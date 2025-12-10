# conformer_decoder.py
import torch
import torch.nn as nn
import torchaudio


class ConformerCTCDecoder(nn.Module):
    """
    Neural-features -> [patch conv] -> torchaudio Conformer -> CTC logits.

    Keeps a compatible API with RNNDecoder but adds an optional `lengths`
    argument so we can pass proper frame lengths into the Conformer.
    """

    def __init__(
        self,
        neuron_capture_tensor_dim: int,
        num_phonemes: int,
        num_days: int,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        ts_patch_size: int = 1,
        ts_patch_stride: int = 1,
    ):
        super().__init__()

        self.input_dim = neuron_capture_tensor_dim
        self.num_phonemes = num_phonemes
        self.d_model = d_model
        self.ts_patch_size = ts_patch_size
        self.ts_patch_stride = ts_patch_stride

        # 1) Patchifying conv: (B, T, C) -> (B, T', d_model)
        self.patch_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.d_model,
            kernel_size=self.ts_patch_size,
            stride=self.ts_patch_stride,
        )

        self.input_layer_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 2) torchaudio Conformer encoder
        # NOTE: according to docs, output has shape (B, T, input_dim)
        self.conformer = torchaudio.models.Conformer(
            input_dim=self.d_model,
            num_heads=num_heads,
            ffn_dim=self.d_model * ff_expansion_factor,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
        )

        # 3) CTC head: project to phoneme logits
        self.out = nn.Linear(self.d_model, self.num_phonemes)

    def forward(self, features, day_indices=None, lengths=None):
        """
        features: (B, T, C) neural features
        day_indices: (B,) or (B,1)  (currently unused, kept for API compatibility)
        lengths: (B,) valid frame counts BEFORE patching, or None.

        returns: logits: (B, T', V)
        """
        # (B, T, C) -> (B, C, T)
        x = features.transpose(1, 2)
        x = self.patch_conv(x)       # (B, d_model, T')
        x = x.transpose(1, 2)        # (B, T', d_model)

        x = self.input_layer_norm(x)
        x = self.input_dropout(x)

        B, T_prime, _ = x.shape

        # If lengths are not provided, assume all frames valid for each example
        if lengths is None:
            conformer_lengths = torch.full(
                (B,),
                T_prime,
                device=x.device,
                dtype=torch.long,
            )
        else:
            # Trainer will pass patch lengths = adjusted_lens (after patching)
            conformer_lengths = lengths

        # torchaudio Conformer expects (B, T, input_dim) and lengths (B,)
        x, out_lengths = self.conformer(x, conformer_lengths)  # x: (B, T', d_model)

        logits = self.out(x)         # (B, T', V)
        return logits
