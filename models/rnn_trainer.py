import json
import math
import pathlib
import pickle
import sys
import time
from omegaconf import OmegaConf
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchaudio.functional as F
import os
import numpy as np
import random
import logging

from rnn_decoder import RNNDecoder
from ctc_beam_search import CTCBeamSearchDecoder

from data_augmentations import gauss_smooth
from dataset import BrainToTextDataset, train_test_split_indicies
from utils.general_utils import aggregate_batch_confidence_metrics, aggregate_phoneme_metrics, \
    compute_confidence_metrics_for_sample, compute_phoneme_metrics, \
    compute_word_char_error

# Configure TF32 using the new API (avoids deprecation warnings in PyTorch 2.9+)
torch.backends.cudnn.conv.fp32_precision = 'tf32'  # TF32 for cuDNN convolution operations
torch.backends.cuda.matmul.fp32_precision = 'tf32'  # TF32 for matrix multiplication on Ampere+ GPUs
torch.backends.cudnn.deterministic = True  # Makes training more reproducible
torch.backends.cudnn.benchmark = False  # Disable auto-tuner for reproducibility
torch._dynamo.config.cache_size_limit = 64  # Limit compilation cache size

class RNNTrainer:
    def __init__(self, args, rank):
        self.args = args
        #------------------------------------------------------------------
        # Setup Logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]: self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')

        #------------------------------------------------------------------
        # Setup Distributed Training
        self.rank = rank
        self.is_distributed = self.rank is not None

        if not self.is_distributed or self.rank == 0:
            if args['mode'] == 'train':
                os.makedirs(self.args['output_dir'], exist_ok = True)
            if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']:
                os.makedirs(self.args['checkpoint_dir'], exist_ok=True)

        if not self.is_distributed or self.rank == 0:
            if args['mode']=='train':
                fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
        
        ## Setup Training Device
        if self.is_distributed:
            dist.init_process_group(backend='nccl', init_method=self.args['dist_url'],
                                    world_size=self.args['world_size'], rank=self.rank)
            self.device = torch.device(f"cuda:{self.rank}")
            torch.cuda.set_device(self.rank)
            self.logger.info(f"Initialized distributed training on rank {self.rank}")
        elif torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0
            
            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0
            
            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # MPS only reachable if no CUDA
            try:
                self.device = torch.device("mps")
                _ = torch.tensor([1.0]).to(self.device)  # test Apple M1/M2/M3 GPU
            except Exception as e:
                self.logger.error(f"Error initializing MPS device: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        
        else:
            # Absolute fallback
            self.device = torch.device("cpu")
        
        if not self.is_distributed or self.rank == 0:
            self.logger.info(f'Using device: {self.device}')
        
        #------------------------------------------------------------------
        #Init Seed
        if self.args['seed'] != -1:
            ## Maintaing Variability accross GPUs
            seed = self.args['seed'] + self.rank if self.is_distributed else self.args['seed']
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        
        #------------------------------------------------------------------
        #Init Model
        self.model = RNNDecoder(
            neuron_capture_tensor_dim = self.args['model']['n_input_features'],
            hidden_state_dim = self.args['model']['n_units'],
            num_days = len(self.args['dataset']['sessions']),
            num_phonemes = self.args['dataset']['n_classes'],
            rnn_type = self.args['model']['rnn_type'],
            rnn_dropout = self.args['model']['rnn_dropout'],
            input_dropout = self.args['model']['input_network']['input_layer_dropout'],
            num_rec_layers = self.args['model']['n_layers'],
            ts_patch_size = self.args['model']['patch_size'],
            ts_patch_stride = self.args['model']['patch_stride'],
            bidirectional = self.args['model']['bidirectional']
        )
        self.model.to(self.device)
        if not self.device == torch.device("mps"):
            if not self.is_distributed or self.rank == 0:
                self.logger.info("Using torch.compile")
                self.model = torch.compile(self.model)
        
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        
        if not self.is_distributed or self.rank == 0:
            self.logger.info(f"Initialized RNN decoding model")
            self.logger.info(self.model)
        
        ## Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        if not self.is_distributed or self.rank == 0:
            self.logger.info(f"Model has {total_params:,} parameters")

        ## Determine how many day-specific parameters are in the model
        day_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()

        if not self.is_distributed or self.rank == 0:
            self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")

        #------------------------------------------------------------------
        #Init Datasets
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in self.args['dataset']['sessions']]

        if len(set(train_file_paths)) != len(train_file_paths): raise ValueError("There are duplicate sessions listed in the train dataset")
        if len(set(val_file_paths)) != len(val_file_paths): raise ValueError("There are duplicate sessions listed in the val dataset")
        
        train_trials, _ = train_test_split_indicies(
            file_paths = train_file_paths,
            test_percentage = 0,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )
        _, val_trials = train_test_split_indicies(
            file_paths = val_file_paths,
            test_percentage = 1,
            seed = self.args['dataset']['seed'],
            bad_trials_dict = None,
            )

        # Save dictionaries to output directory to know which trials were train vs val
        if not self.is_distributed or self.rank == 0:
            with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f:
                json.dump({'train' : train_trials, 'val': val_trials}, f)

        # Determine if a only a subset of neural features should be used
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] != None:
            feature_subset = self.args['dataset']['feature_subset']
            if not self.is_distributed or self.rank == 0:
                self.logger.info(f'Using only a subset of features: {feature_subset}')

        # train dataset and dataloader
        self.train_dataset = BrainToTextDataset(
            trial_indicies = train_trials,
            split = 'train',
            days_per_batch = self.args['dataset']['days_per_batch'],
            n_batches = self.args['num_training_batches'],
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
            )
        train_sampler = None
        shuffle = self.args['dataset']['loader_shuffle']
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True, seed=self.args['seed'])
            shuffle = False # Sampler handles shuffling

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = shuffle,
            num_workers = self.args['dataset']['num_dataloader_workers'],
            pin_memory = True,
            sampler=train_sampler
        )

        # val dataset and dataloader
        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials,
            split = 'test',
            days_per_batch = None,
            n_batches = None,
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
            )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = False,
            num_workers = 0,
            pin_memory = True
        )

        if not self.is_distributed or self.rank == 0:
            self.logger.info("Successfully initialized datasets")

        #------------------------------------------------------------------
        #Init Optimizer
        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer = self.optimizer,
                start_factor = 1.0,
                end_factor = self.args['lr_min']/ self.args['lr_max'],
                total_iters = self.args['lr_decay_steps']
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")

        #------------------------------------------------------------------
        #Init Loss
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction='none', zero_infinity=False)

        #------------------------------------------------------------------
        #Init Beam Search Decoder (optional)
        self.use_beam_search = self.args.get('use_beam_search', False)
        if self.use_beam_search:
            self.beam_width = self.args.get('beam_width', 10)
            prune_threshold = self.args.get('beam_prune_threshold', -10.0)
            verbose = self.args.get('beam_search_logging', False)
            self.beam_search_decoder = CTCBeamSearchDecoder(
                blank_id=0,
                beam_width=self.beam_width,
                prune_threshold=prune_threshold,
                verbose = verbose
            )

            # Share logger handlers with beam search decoder
            beam_logger = logging.getLogger('ctc_beam_search')
            for handler in self.logger.handlers:
                beam_logger.addHandler(handler)
            beam_logger.setLevel(logging.INFO)

            if not self.is_distributed or self.rank == 0:
                self.logger.info(f"Using beam search decoding with beam_width={self.beam_width}, prune_threshold={prune_threshold}")
        else:
            if not self.is_distributed or self.rank == 0:
                self.logger.info("Using greedy decoding")

        #------------------------------------------------------------------
        #Load from checkpoint
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        #------------------------------------------------------------------
        # Set rnn and/or input layers to not trainable if specified
        for name, param in self.model.named_parameters():
            if not self.args['model']['rnn_trainable'] and 'rnn' in name:
                param.requires_grad = False

            elif not self.args['model']['input_network']['input_trainable'] and 'day' in name:
                param.requires_grad = False
        
        #------------------------------------------------------------------
        # Load data transform args
        self.transform_args = self.args['dataset']['data_transforms']

        #------------------------------------------------------------------
        # Initialize best validation PER & Loss
        self.best_val_PER = torch.inf
        self.best_val_loss = torch.inf
    
    def create_optimizer(self):
        '''
        we choose not to decay biases and day wegihts 
        day weights should have a seperate learning rate
        '''
        bias_params = [p for name, p in self.model.named_parameters() if 'rnn.bias' in name or 'out.bias' in name]
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and 'rnn.bias' not in name and 'out.bias' not in name]

        if len(day_params) != 0:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        
        optim = torch.optim.AdamW(
            param_groups,
            lr = self.args['lr_max'],
            betas = (self.args['beta0'], self.args['beta1']),
            eps = self.args['epsilon'],
            weight_decay = self.args['weight_decay'],
            fused = True
        )

        return optim
    
    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio)*(cosine_decay))
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps), # biases
                lambda step: lr_lambda(
                    step,
                    lr_min_day / lr_max_day,
                    lr_decay_steps_day,
                    lr_warmup_steps_day,
                    ), # day params
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps), # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps), # biases
                lambda step: lr_lambda(
                    step,
                    lr_min / lr_max,
                    lr_decay_steps,
                    lr_warmup_steps), # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")

        return LambdaLR(optim, lr_lambdas)
    
    def transform_data(self, features, n_time_steps, mode = 'train'):
        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        if mode == 'train':
            # add static gain noise
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features,
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )


        return features, n_time_steps

    def validation(self, loader, return_logits = False, return_data = False):
        self.model.eval()
        metrics = {
            'decoded_seqs': [],
            'true_seq': [],
            'phone_seq_lens': [],
            'transcription': [],
            'losses': [],
            'block_nums': [],
            'trial_nums': [],
            'day_indicies': [],
            # NEW
            'confidence_samples': [],
            'confidence_batches': []
        }

        if return_logits:
            metrics['logits'] = []
            metrics['n_time_steps'] = []
        
        if return_data:
            metrics['input_features'] = []
        
        total_edit_distance = 0
        total_seq_len = 0

        day_per = {}
        for d in range(len(self.args['dataset']['sessions'])):
            if self.args['dataset']['dataset_probability_val'][d] == 1:
                day_per[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}

        for _, batch in enumerate(loader):
            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            day = day_indicies[0].item() # validation batches are day specific
            if self.args['dataset']['dataset_probability_val'][day] == 0:
                if self.args['log_val_skip_logs']:
                    self.logger.info(f"Skipping validation on day {day}")
                continue
                
            with torch.no_grad():
                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')

                    adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model']['patch_stride'] + 1).to(torch.int32)

                    logits = self.model(features, day_indicies)
                    
                    # Try CUDA/MPS direct CTC loss first
                    try:
                        loss = self.ctc_loss(
                            log_probs=torch.permute(logits.log_softmax(2), (1, 0, 2)),
                            targets=labels,
                            input_lengths=adjusted_lens,
                            target_lengths=phone_seq_lens
                        )
                    except Exception:
                        # Fallback for MPS
                        logits_cpu = logits.detach().cpu().requires_grad_()
                        loss = self.ctc_loss(
                            log_probs=torch.permute(logits_cpu.log_softmax(2), (1, 0, 2)),
                            targets=labels.cpu(),
                            input_lengths=adjusted_lens.cpu(),
                            target_lengths=phone_seq_lens.cpu()
                        )
                    
                    loss = torch.mean(loss)

                metrics['losses'].append(loss.cpu().detach().numpy())

                batch_edit_distance = 0
                decoded_seqs = []

                # Decode sequences (greedy or beam search)
                if self.use_beam_search:
                    # Beam search decoding - use parallel processing for large batches
                    use_parallel = logits.shape[0] >= 8  # Enable for batches of 8+
                    decoded_seqs_batch = self.beam_search_decoder.decode_batch(logits, adjusted_lens, parallel=use_parallel)

                    for iterIdx in range(logits.shape[0]):
                        decoded_seq = decoded_seqs_batch[iterIdx]
                        trueSeq = np.array(
                            labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach()
                        )

                        batch_edit_distance += F.edit_distance(decoded_seq, trueSeq)
                        decoded_seqs.append(decoded_seq)
                    # Beam search does NOT produce per-time-step logits,
                    # so we do NOT compute confidence metrics here.
                    sample_conf_metrics = None
                else:
                    # Greedy decoding
                    for iterIdx in range(logits.shape[0]):
                        decoded_seq = torch.argmax(logits[iterIdx, 0 : adjusted_lens[iterIdx], :], dim=-1)
                        decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                        decoded_seq = decoded_seq.cpu().detach().numpy()
                        decoded_seq = np.array([i for i in decoded_seq if i != 0])

                        trueSeq = np.array(
                            labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach()
                        )

                        batch_edit_distance += F.edit_distance(decoded_seq, trueSeq)

                        decoded_seqs.append(decoded_seq)
                        
                        # --------------------------------------------
                        # NEW: phoneme confidence metrics (only greedy)
                        # --------------------------------------------
                        if self.args['metrics']['activation']:
                            logits_i = logits[iterIdx, :adjusted_lens[iterIdx], :].detach()
                            conf_m = compute_confidence_metrics_for_sample(
                                logits_tensor=logits_i,
                                adjusted_len=int(adjusted_lens[iterIdx].item()),
                                true_phonemes=trueSeq,
                                blank_id=0
                            )
                            sample_conf_metrics.append(conf_m)  # Only greedy supports it
                        else:
                            sample_conf_metrics = None

            day = batch['day_indicies'][0].item()

            day_per[day]['total_edit_distance'] += batch_edit_distance
            day_per[day]['total_seq_length'] += torch.sum(phone_seq_lens).item()

            total_edit_distance += batch_edit_distance
            total_seq_len += torch.sum(phone_seq_lens)

            if return_logits:
                metrics['logits'].append(logits.cpu().float().numpy()) # Will be in bfloat16 if AMP is enabled, so need to set back to float32
                metrics['n_time_steps'].append(adjusted_lens.cpu().numpy())

            if return_data:
                metrics['input_features'].append(batch['input_features'].cpu().numpy())

            metrics['decoded_seqs'].append(decoded_seqs)
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].numpy())
            metrics['trial_nums'].append(batch['trial_nums'].numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

        avg_PER = total_edit_distance / total_seq_len

        metrics['day_PERs'] = day_per
        metrics['avg_PER'] = avg_PER.item()
        metrics['avg_loss'] = np.mean(metrics['losses'])

        return metrics

    def load_model_checkpoint(self, load_path):
        '''
        Load a training checkpoint
        '''
        checkpoint = torch.load(load_path, weights_only = False) # checkpoint is just a dict

        model_to_load = self.model.module if self.is_distributed else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER'] # best phoneme error rate
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf

        self.model.to(self.device)

        # Send optimizer params back to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        if not self.is_distributed or self.rank == 0:
            self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, best_PER, best_loss):
        '''
        Save a training checkpoint
        '''

        model_state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        checkpoint = {
            'model_state_dict' : model_state_dict,
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.learning_rate_scheduler.state_dict(),
            'val_PER' : best_PER,
            'val_loss' : best_loss
        }

        torch.save(checkpoint, save_path)

        if not self.is_distributed or self.rank == 0:
            self.logger.info("Saved model to checkpoint: " + save_path)

            # Save the args file alongside the checkpoint
            with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
                OmegaConf.save(config=self.args, f=f)
                
    def train(self):
        #------------------------------------------------------------------
        # performance tracking
        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []

        val_steps_since_improvement = 0

        #------------------------------------------------------------------
        # training params
        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)
        early_stopping_val_steps = self.args['early_stopping_val_steps']

        #------------------------------------------------------------------
        # Start Training
        train_start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            # this is needed to select a different training batch for each epoch
            if self.is_distributed: self.train_loader.sampler.set_epoch(i)
            #------------------------------------------------------------------
            # Set model to train mode: Enables Dropout / Running Averages on Norms etc.
            self.model.train()
            self.optimizer.zero_grad()

            epoch_start_time = time.time()

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indices = batch['day_indicies'].to(self.device)

            with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.bfloat16):
                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')
                # As we squash time steps into windows during model forward pass
                adjusted_lens = ((n_time_steps - self.args['model']['patch_size']) / self.args['model']['patch_stride'] + 1).to(torch.int32)
                logits = self.model(features, day_indices)
                try:
                    loss = self.ctc_loss(
                        log_probs=torch.permute(logits.log_softmax(2), (1, 0, 2)),  # expected input dimension (T, N, S)
                        targets=labels,
                        input_lengths=adjusted_lens,
                        target_lengths=phone_seq_lens
                    )
                except Exception as e:
                    # not supported for MPS
                    logits_cpu = logits.detach().cpu().requires_grad_()
                    
                    log_probs_cpu = torch.permute(
                        logits_cpu.log_softmax(2), (1, 0, 2)
                    )
                    
                    loss = self.ctc_loss(
                        log_probs=log_probs_cpu,
                        targets=labels.cpu(),
                        input_lengths=adjusted_lens.cpu(),
                        target_lengths=phone_seq_lens.cpu()
                    )
            
            loss = torch.mean(loss)
            
            loss.backward()

            #------------------------------------------------------------------
            # Clip Gradients
            if self.args['grad_norm_clip_value'] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                max_norm = self.args['grad_norm_clip_value'],
                                error_if_nonfinite = True,
                                foreach=True
                            )
            
            self.optimizer.step()
            self.learning_rate_scheduler.step()

            train_step_duration = time.time() - epoch_start_time
            train_losses.append(loss.detach().item())
            # Train Step Complete
            #-----------------------------------------------------------------

            # Logging
            if i % self.args['batches_per_train_log'] == 0 and (not self.is_distributed or self.rank == 0):
                self.logger.info(f'Train batch {i}: ' +
                    f'loss: {(loss.detach().item()):.2f} ' +
                    f'grad norm: {grad_norm:.2f} ' +
                    f'time: {train_step_duration:.3f}')

            #-----------------------------------------------------------------
            # Validation
            if (i % self.args['batches_per_val_step'] == 0 or i == ((self.args['num_training_batches'] - 1))) and (not self.is_distributed or self.rank == 0):
                self.logger.info(f"Running validation after training batch: {i}")

                val_start_time = time.time()
                val_metrics = self.validation(loader = self.val_loader, return_logits = self.args['save_val_logits'], return_data = self.args['save_val_data'])
                val_step_duration = time.time() - val_start_time

                self.logger.info(f'Val batch {i}: ' +
                    f'PER (avg): {val_metrics["avg_PER"]:.4f} ' +
                    f'CTC Loss (avg): {val_metrics["avg_loss"]:.4f} ' +
                    f'time: {val_step_duration:.3f}')
                
                if self.args['log_individual_day_val_PER']:
                    for day in val_metrics['day_PERs'].keys():
                        self.logger.info(f"{self.args['dataset']['sessions'][day]} val PER: {val_metrics['day_PERs'][day]['total_edit_distance'] / val_metrics['day_PERs'][day]['total_seq_length']:0.4f}")
                
                val_PERs.append(val_metrics['avg_PER'])
                val_losses.append(val_metrics['avg_loss'])
                val_results.append(val_metrics)

                new_best = False
                if val_metrics['avg_PER'] < self.best_val_PER:
                    self.logger.info(f"New best test PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                    self.best_val_PER = val_metrics['avg_PER']
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True
                elif val_metrics['avg_PER'] == self.best_val_PER and (val_metrics['avg_loss'] < self.best_val_loss):
                    self.logger.info(f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True

                if new_best:
                    # Checkpoint if metrics have improved
                    if save_best_checkpoint:
                        self.logger.info(f"Checkpointing model")
                        self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/best_checkpoint', self.best_val_PER, self.best_val_loss)
                    # save validation metrics to pickle file
                    if self.args['save_val_metrics']:
                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', 'wb') as f:
                            pickle.dump(val_metrics, f)
                    val_steps_since_improvement = 0
                else:
                    val_steps_since_improvement +=1

                # Optionally save this validation checkpoint, regardless of performance
                if self.args['save_all_val_steps']:
                    self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}', val_metrics['avg_PER'])

                # Early stopping
                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(f'Overall validation PER has not improved in {early_stopping_val_steps} validation steps. Stopping training early at batch: {i}')
                    break
        
        training_duration = time.time() - train_start_time

        if not self.is_distributed or self.rank == 0:
            self.logger.info(f'Best avg val PER achieved: {self.best_val_PER:.5f}')
            self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        # Save final model
        if self.args['save_final_model'] and (not self.is_distributed or self.rank == 0):
            self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}', val_PERs[-1])

        train_stats = {}
        train_stats['train_losses'] = train_losses
        train_stats['val_losses'] = val_losses
        train_stats['val_PERs'] = val_PERs
        train_stats['val_metrics'] = val_results

        if self.is_distributed:
            dist.destroy_process_group()

        return train_stats
