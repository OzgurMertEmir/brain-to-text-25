import torch
from torch import nn

class BaseDecoder(nn.Module):
    def __init__(
        self,
        neuron_capture_tensor_dim,
        num_days,
        input_dropout=0.0,
        ts_patch_size=0,
        ts_patch_stride=0,
    ):
        super(BaseDecoder, self).__init__()
        self.neuron_capture_tensor_dim = neuron_capture_tensor_dim
        self.num_days = num_days
        self.input_dropout = input_dropout
        self.ts_patch_size = ts_patch_size
        self.ts_patch_stride = ts_patch_stride

        self.day_weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.eye(self.neuron_capture_tensor_dim)
                ) for _ in range(self.num_days)
            ]
        )
        self.day_biases = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(1, self.neuron_capture_tensor_dim)
                ) for _ in range(self.num_days)
            ]
        )
        self.day_layer_activation = nn.Softsign()
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neuron_capture_tensor_dim * self.ts_patch_size if self.ts_patch_size > 0 else self.neuron_capture_tensor_dim

    def forward(self, x, day_idx):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        '''

        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.stack([self.day_biases[i] for i in day_idx], dim=0)

        x = self.day_layer_activation(torch.matmul(x, day_weights) + day_biases)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        if self.ts_patch_size > 0:
            x = x.unfold(1, self.ts_patch_size, self.ts_patch_stride).permute(0, 1, 3, 2)
            x = x.reshape(x.size(0), x.size(1), -1)
        
        return x