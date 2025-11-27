import torch
from torch import nn
from base_decoder import BaseDecoder

class RNNDecoder(BaseDecoder):
    def __init__(
        self,
        neuron_capture_tensor_dim,
        hidden_state_dim,
        num_days,
        num_phonemes,
        rnn_type='GRU',  # RNN, LSTM, or GRU
        rnn_dropout=0.0,
        input_dropout=0.0,
        num_rec_layers=5,
        ts_patch_size=0,
        ts_patch_stride=0,
        bidirectional=False
    ):
        super(RNNDecoder, self).__init__(
            neuron_capture_tensor_dim=neuron_capture_tensor_dim,
            num_days=num_days,
            input_dropout=input_dropout,
            ts_patch_size=ts_patch_size,
            ts_patch_stride=ts_patch_stride,
        )
        self.hidden_state_dim = hidden_state_dim
        self.num_phonemes = num_phonemes
        self.num_rec_layers = num_rec_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout

        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_rec_layers * (2 if self.bidirectional else 1), 1, self.hidden_state_dim)))
        if self.rnn_type == 'LSTM':
            self.c0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_rec_layers * (2 if self.bidirectional else 1), 1, self.hidden_state_dim)))

        rnn_class = nn.RNN if rnn_type == 'RNN' else (nn.LSTM if rnn_type == 'LSTM' else nn.GRU)
        
        self.rnn = rnn_class(
            input_size=self.input_size,
            hidden_size=self.hidden_state_dim,
            num_layers=self.num_rec_layers,
            dropout=self.rnn_dropout,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
        
        self.out = nn.Linear(self.hidden_state_dim * (2 if self.bidirectional else 1), self.num_phonemes)
        nn.init.xavier_uniform_(self.out.weight)
    
    def forward(self, x, day_idx, states=None, return_state=False, return_embedding=False):
        x = super().forward(x, day_idx)
        
        if states is None:
            h0 = self.h0.expand(self.num_rec_layers * (2 if self.bidirectional else 1), x.shape[0], self.hidden_state_dim).contiguous()
            if self.rnn_type == 'LSTM':
                c0 = self.c0.expand(self.num_rec_layers * (2 if self.bidirectional else 1), x.shape[0], self.hidden_state_dim).contiguous()
                states = (h0, c0)
            else:
                states = h0

        output, hidden_states = self.rnn(x, states)
        logits = self.out(output)

        if return_state: return logits, hidden_states
        elif return_embedding: return logits, output
        return logits
