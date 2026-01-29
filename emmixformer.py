import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class PreprocessingBlock(nn.Module):
    """
    Preprocess raw eye-movement sequences into slow/fast streams.

    Expected input shapes (x):
      - (batch, time, 2)  -> will be transposed internally to (batch, 2, time)
      - (batch, 2, time)

    The block:
      - computes velocity magnitude between consecutive samples
      - thresholds velocity (default 40 deg/s) into:
          slow  : likely fixations / smooth pursuit
          fast  : likely saccades
      - returns tensors shaped for Conv1d: (batch, 2, time)
        with masking (zeros) applied per stream.
    """

    def __init__(self, velocity_threshold: float = 40.0, eps: float = 1e-6):
        super().__init__()
        self.velocity_threshold = velocity_threshold
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure shape is (B, 2, T)
        if x.dim() != 3:
            raise ValueError(f"PreprocessingBlock expects 3D input, got shape {x.shape}")

        if x.size(1) == 2:
            coords = x  # (B, 2, T)
        elif x.size(2) == 2:
            coords = x.transpose(1, 2)  # (B, T, 2) -> (B, 2, T)
        else:
            raise ValueError(
                "PreprocessingBlock expects input with 2 coordinate channels "
                f"in either dim 1 or 2, got shape {x.shape}"
            )

        # Approximate velocity as first difference along time
        # v[:, :, 0] == 0 to keep same temporal length
        dx = coords[:, :, 1:] - coords[:, :, :-1]  # (B, 2, T-1)
        v = torch.zeros_like(coords)
        v[:, :, 1:] = dx

        # Velocity magnitude (assuming coordinates already in deg or appropriately scaled)
        vel_mag = torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + self.eps)  # (B, T)

        # Create slow / fast masks; keep dimensions broadcastable to (B, 2, T)
        slow_mask = (vel_mag < self.velocity_threshold).unsqueeze(1)  # (B, 1, T)
        fast_mask = ~slow_mask  # (B, 1, T)

        slow_stream = coords * slow_mask  # (B, 2, T)
        fast_stream = coords * fast_mask  # (B, 2, T)

        return slow_stream, fast_stream


class ConvBlock1D(nn.Module):
    """
    Basic 1D convolutional block:
      Conv1d -> BatchNorm1d -> ReLU -> AvgPool1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_kernel_size: int = 2,
    ):
        super().__init__()
        padding = kernel_size // 2  # keep temporal length before pooling
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SiameseCNN(nn.Module):
    """
    Siamese CNN with two identical branches processing:
      - slow stream
      - fast stream

    Each branch:
      4x [Conv1d -> BN -> ReLU -> AvgPool1d],
      with increasing kernel sizes to expand receptive field.

    Input shapes:
      slow, fast: (B, 2, T)

    Output:
      concatenated feature map: (B, 2 * out_channels_last_block, T_out)
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        kernel_sizes: Tuple[int, int, int, int] = (3, 5, 7, 9),
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        channels = [in_channels, c1, c2, c3, c4]

        self.branch = nn.Sequential(
            ConvBlock1D(channels[0], channels[1], kernel_sizes[0]),
            ConvBlock1D(channels[1], channels[2], kernel_sizes[1]),
            ConvBlock1D(channels[2], channels[3], kernel_sizes[2]),
            ConvBlock1D(channels[3], channels[4], kernel_sizes[3]),
        )

        self.out_channels = channels[-1]

    def forward(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        if slow.shape != fast.shape:
            raise ValueError(
                f"SiameseCNN expects slow/fast tensors of same shape, "
                f"got {slow.shape} and {fast.shape}"
            )

        slow_feat = self.branch(slow)
        fast_feat = self.branch(fast)

        # Concatenate along channel dimension
        out = torch.cat([slow_feat, fast_feat], dim=1)
        return out  # (B, 2 * out_channels, T_out)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, d_model)
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class AttentiveLSTMCell(nn.Module):
    """
    Custom LSTM cell with a simple attention mechanism and peephole connections.

    At each time step:
      - computes a scalar attention weight between x_t and h_{t-1}
      - forms a context vector as a mixture of x_t and h_{t-1}
      - uses peephole connections (c_{t-1} into gates)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters for attention over (x_t, h_{t-1})
        self.W_q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_proj = nn.Linear(input_size + hidden_size, input_size, bias=False)

        # LSTM gates with peephole connections
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self.w_ci = nn.Parameter(torch.zeros(hidden_size))
        self.w_cf = nn.Parameter(torch.zeros(hidden_size))
        self.w_co = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_t: (B, input_size), h_prev, c_prev: (B, hidden_size)

        # Attention between x_t and h_prev
        q = self.W_q(x_t)  # (B, H)
        k = self.W_k(h_prev)  # (B, H)
        v = self.W_v(h_prev)  # (B, H)

        # Scaled dot-product attention, single score per batch element
        attn_scores = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        alpha = torch.sigmoid(attn_scores)  # (B, 1), in (0,1)

        context = alpha * v + (1.0 - alpha) * x_t  # broadcast over feature dim

        # Optionally project concatenation for richer interaction
        mixed = torch.cat([context, h_prev], dim=-1)  # (B, input+H)
        x_tilde = self.attn_proj(mixed)  # (B, input_size)

        # LSTM with peephole connections
        gates = self.W_x(x_tilde) + self.W_h(h_prev)  # (B, 4H)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i + self.w_ci * c_prev)
        f = torch.sigmoid(f + self.w_cf * c_prev)
        g = torch.tanh(g)
        c_t = f * c_prev + i * g
        o = torch.sigmoid(o + self.w_co * c_t)
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class AttentionLSTM(nn.Module):
    """
    Sequence wrapper around AttentiveLSTMCell.

    Input:
      x: (B, T, input_size)

    Output:
      h_seq: (B, T, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = AttentiveLSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = x.new_zeros(B, self.cell.hidden_size)
        c = x.new_zeros(B, self.cell.hidden_size)

        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # (B, T, H)


class StandardTransformerEncoder(nn.Module):
    """
    Wrapper around nn.TransformerEncoder with positional encoding.

    Input:
      x: (B, T, d_model)
    Output:
      x_enc: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # we'll transform to (T, B, C)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.transpose(0, 1)  # (T, B, C)
        x = self.pos_encoder(x)
        x = self.encoder(x)  # (T, B, C)
        x = x.transpose(0, 1)  # (B, T, C)
        return x


class FourierTransformer(nn.Module):
    """
    Fourier Transformer module:
      - applies 1D FFT along the temporal dimension
      - splits complex spectrum into amplitude & phase
      - processes each with its own Transformer encoder
      - recombines them and applies inverse FFT to return to time domain

    Input:
      x: (B, T, d_model)
    Output:
      x_time: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.amp_transformer = StandardTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.phase_transformer = StandardTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        # FFT along time dimension
        X_f = torch.fft.rfft(x, dim=1)  # (B, F, C), F = T//2+1

        amplitude = torch.abs(X_f)  # (B, F, C)
        phase = torch.angle(X_f)  # (B, F, C)

        # Process amplitude and phase as sequences over frequency
        amp_in = amplitude  # (B, F, C)
        phase_in = phase  # (B, F, C)

        amp_out = self.amp_transformer(amp_in)  # (B, F, C)
        phase_out = self.phase_transformer(phase_in)  # (B, F, C)

        # Recombine
        complex_spec = amp_out * torch.exp(1j * phase_out)  # (B, F, C)

        # Inverse FFT back to time domain
        x_time = torch.fft.irfft(complex_spec, n=T, dim=1)  # (B, T, C)
        return x_time


class MixBlock(nn.Module):
    """
    Mixed block integrating:
      - Attention LSTM (short-term dependencies)
      - Standard Transformer (long-range dependencies)
      - Fourier Transformer (frequency-domain global features)

    Input:
      x: (B, C_in, T)

    Output:
      concatenated features: (B, C_out, T)
        where C_out = att_hidden_dim + trans_dim + fourier_dim
    """

    def __init__(
        self,
        in_channels: int,
        att_hidden_dim: int = 128,
        trans_dim: int = 128,
        fourier_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project input channels to each module's working dimension
        self.proj_att = nn.Linear(in_channels, att_hidden_dim)
        self.proj_trans = nn.Linear(in_channels, trans_dim)
        self.proj_fourier = nn.Linear(in_channels, fourier_dim)

        self.att_lstm = AttentionLSTM(att_hidden_dim, att_hidden_dim)
        self.transformer = StandardTransformerEncoder(
            d_model=trans_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fourier_transformer = FourierTransformer(
            d_model=fourier_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.out_channels = att_hidden_dim + trans_dim + fourier_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        B, C, T = x.shape
        x_t = x.transpose(1, 2)  # (B, T, C)

        # Attention LSTM branch
        att_in = self.proj_att(x_t)  # (B, T, att_hidden_dim)
        att_out = self.att_lstm(att_in)  # (B, T, att_hidden_dim)

        # Standard Transformer branch
        trans_in = self.proj_trans(x_t)  # (B, T, trans_dim)
        trans_out = self.transformer(trans_in)  # (B, T, trans_dim)

        # Fourier Transformer branch
        fourier_in = self.proj_fourier(x_t)  # (B, T, fourier_dim)
        fourier_out = self.fourier_transformer(fourier_in)  # (B, T, fourier_dim)

        # Concatenate along feature dimension
        mixed = torch.cat([att_out, trans_out, fourier_out], dim=-1)  # (B, T, C_out)
        mixed = mixed.transpose(1, 2)  # (B, C_out, T)
        return mixed


class EmMixformer(nn.Module):
    """
    Full EmMixformer model:
      PreprocessingBlock -> SiameseCNN -> MixBlock -> ClassifierHead
    """

    def __init__(
        self,
        num_classes: int,
        velocity_threshold: float = 40.0,
        cnn_base_channels: int = 32,
        mix_att_hidden_dim: int = 128,
        mix_trans_dim: int = 128,
        mix_fourier_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.preprocess = PreprocessingBlock(velocity_threshold=velocity_threshold)
        self.siamese_cnn = SiameseCNN(
            in_channels=2,
            base_channels=cnn_base_channels,
        )

        mix_in_channels = self.siamese_cnn.out_channels * 2

        self.mix_block = MixBlock(
            in_channels=mix_in_channels,
            att_hidden_dim=mix_att_hidden_dim,
            trans_dim=mix_trans_dim,
            fourier_dim=mix_fourier_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
        )

        self.feature_dim = self.mix_block.out_channels
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw eye-movement coordinates
               shape (B, T, 2) or (B, 2, T)

        Returns:
            logits: (B, num_classes)
        """
        slow, fast = self.preprocess(x)  # (B, 2, T), (B, 2, T)
        cnn_out = self.siamese_cnn(slow, fast)  # (B, C_cnn, T_cnn)

        mixed = self.mix_block(cnn_out)  # (B, C_mix, T_cnn)

        # Global average pooling over time
        feat = mixed.mean(dim=-1)  # (B, C_mix)
        logits = self.classifier(feat)  # (B, num_classes)
        return logits


if __name__ == "__main__":
    # Simple sanity check with random input
    batch_size = 4
    seq_len = 512
    num_classes = 10

    model = EmMixformer(num_classes=num_classes)
    dummy_input = torch.randn(batch_size, seq_len, 2)  # (B, T, 2)
    out = model(dummy_input)
    print("Output shape:", out.shape)  # should be (4, 10)

