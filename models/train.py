import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
import os

def main_worker(rank, args):
    """
    Main worker function for distributed training.
    """
    trainer = BrainToTextDecoder_Trainer(args, rank)
    trainer.train()

if __name__ == '__main__':
    args = OmegaConf.load('args.yaml')

    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        args.dist_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        args.world_size = torch.cuda.device_count()
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        trainer = BrainToTextDecoder_Trainer(args, None)
        metrics = trainer.train()