# conformer_decoder.py
import torch
import torch.nn as nn
import torchaudio


class ConformerCTCDecoder(nn.Module):
    """
    features -> day/session affine -> patch_conv -> torchaudio Conformer -> CTC logits
    """

    def __init__(
        self,
        neuron_capture_tensor_dim: int,
        num_phonemes: int,
        num_days: int,
        input_dropout: float = 0.0,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        ts_patch_size: int = 1,
        ts_patch_stride: int = 1,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.input_dim = int(neuron_capture_tensor_dim)
        self.num_phonemes = int(num_phonemes)
        self.num_days = int(num_days)

        self.d_model = int(d_model)
        self.ts_patch_size = int(ts_patch_size)
        self.ts_patch_stride = int(ts_patch_stride)

        # --- Day/session adaptation ---
        # day_weights: (D, C, C), day_biases: (D, 1, C)
        eye = torch.eye(self.input_dim)
        self.day_weights = nn.Parameter(eye.unsqueeze(0).repeat(self.num_days, 1, 1))
        self.day_biases = nn.Parameter(torch.zeros(self.num_days, 1, self.input_dim))

        self.day_layer_activation = nn.Softsign()
        self.day_input_dropout_p = float(input_dropout)
        self.day_layer_dropout = nn.Dropout(self.day_input_dropout_p)

        # --- Patchifying conv: (B, T, C) -> (B, T', d_model) ---
        self.patch_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.d_model,
            kernel_size=self.ts_patch_size,
            stride=self.ts_patch_stride,
        )

        self.input_layer_norm = nn.LayerNorm(self.d_model)
        self.model_dropout = nn.Dropout(dropout)

        # --- torchaudio Conformer ---
        self.conformer = torchaudio.models.Conformer(
            input_dim=self.d_model,
            num_heads=num_heads,
            ffn_dim=self.d_model * ff_expansion_factor,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first,
        )

        # --- CTC head ---
        self.out = nn.Linear(self.d_model, self.num_phonemes)

    def forward(self, features, day_indices=None, lengths=None):
        """
        features: (B, T, C)
        day_indices: (B,) or (B,1) -> required (we apply day adaptation)
        lengths: (B,) length AFTER patching (i.e., adjusted_lens), or None
        """
        if day_indices is None:
            raise ValueError("day_indices is required for day/session adaptation")

        x = features
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected features (B,T,{self.input_dim}), got {tuple(x.shape)}")

        # Ensure shape (B,)
        if day_indices.dim() > 1:
            day_indices = day_indices.view(-1)
        day_indices = day_indices.long()

        # Gather per-example day weights/biases
        # W: (B, C, C), b: (B, 1, C)
        W = self.day_weights.index_select(0, day_indices)
        b = self.day_biases.index_select(0, day_indices)

        # Apply affine + Softsign: (B,T,C) @ (B,C,C) -> (B,T,C)
        x = torch.bmm(x, W) + b
        x = self.day_layer_activation(x)

        if self.day_input_dropout_p > 0.0:
            x = self.day_layer_dropout(x)

        # Patch conv expects (B, C, T)
        x = x.transpose(1, 2)          # (B, C, T)
        x = self.patch_conv(x)         # (B, d_model, T')
        x = x.transpose(1, 2)          # (B, T', d_model)

        x = self.input_layer_norm(x)
        x = self.model_dropout(x)

        B, T_prime, _ = x.shape

        if lengths is None:
            conformer_lengths = torch.full((B,), T_prime, device=x.device, dtype=torch.long)
        else:
            conformer_lengths = lengths.view(-1).long()

        x, out_lengths = self.conformer(x, conformer_lengths)  # (B, T', d_model)
        logits = self.out(x)                                   # (B, T', V)
        return logits