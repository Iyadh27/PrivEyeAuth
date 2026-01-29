import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class PreprocessingBlock(nn.Module):
    """
    Preprocess raw eye-movement sequences into slow/fast streams.
    
    According to the EmMixformer paper (Section III-B), the preprocessing splits
    eye movement data based on velocity threshold (default 40°/s):
    - Slow data: velocities < 40°/s (fixations/smooth pursuit)
    - Fast data: velocities >= 40°/s (saccades)
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

        # Compute velocity as first difference
        dx = coords[:, :, 1:] - coords[:, :, :-1]  # (B, 2, T-1)
        v = torch.zeros_like(coords)
        v[:, :, 1:] = dx
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        # Velocity magnitude
        vel_mag = torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + self.eps)  # (B, T)
        vel_mag = torch.nan_to_num(vel_mag, nan=0.0, posinf=0.0, neginf=0.0)

        # Split into slow/fast streams
        slow_mask = (vel_mag < self.velocity_threshold).unsqueeze(1)  # (B, 1, T)
        fast_mask = ~slow_mask

        slow_stream = coords * slow_mask  # (B, 2, T)
        fast_stream = coords * fast_mask  # (B, 2, T)

        return slow_stream, fast_stream


class ConvBlock1D(nn.Module):
    """
    Basic 1D convolutional block with proper initialization:
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
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SiameseCNN(nn.Module):
    """
    Siamese CNN as described in EmMixformer paper (Section II-A).
    Two identical branches process slow and fast streams with 4 convolutional blocks.
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
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
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
    Attention LSTM cell as described in EmMixformer paper (Section II-B-b).
    Incorporates attention mechanism into LSTM with peephole connections.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Attention parameters
        self.W_q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_proj = nn.Linear(input_size + hidden_size, input_size, bias=False)

        # LSTM gates
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        # Peephole connections
        self.w_ci = nn.Parameter(torch.zeros(hidden_size))
        self.w_cf = nn.Parameter(torch.zeros(hidden_size))
        self.w_co = nn.Parameter(torch.zeros(hidden_size))
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention mechanism
        q = self.W_q(x_t)
        k = self.W_k(h_prev)
        v = self.W_v(h_prev)

        attn_scores = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        alpha = torch.sigmoid(attn_scores)

        context = alpha * v + (1.0 - alpha) * x_t

        mixed = torch.cat([context, h_prev], dim=-1)
        x_tilde = self.attn_proj(mixed)

        # LSTM gates with peephole connections
        gates_x = self.W_x(x_tilde)
        gates_h = self.W_h(h_prev)
        
        i_x, f_x, o_x, g_x = gates_x.chunk(4, dim=-1)
        i_h, f_h, o_h, g_h = gates_h.chunk(4, dim=-1)

        i = torch.sigmoid(i_x + i_h + self.w_ci * c_prev)
        f = torch.sigmoid(f_x + f_h + self.w_cf * c_prev)
        g = torch.tanh(g_x + g_h)
        
        c = f * c_prev + i * g
        o = torch.sigmoid(o_x + o_h + self.w_co * c)
        h = o * torch.tanh(c)

        return h, c


class AttentionLSTM(nn.Module):
    """
    Multi-layer Attention LSTM as described in the paper.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList([
            AttentiveLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_size)
        batch_size, seq_len, _ = x.shape
        device = x.device

        outputs = []
        for layer_idx, cell in enumerate(self.cells):
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
            
            layer_outputs = []
            for t in range(seq_len):
                if layer_idx == 0:
                    x_t = x[:, t, :]
                else:
                    x_t = outputs[-1][:, t, :]
                h, c = cell(x_t, h, c)
                layer_outputs.append(h.unsqueeze(1))
            
            outputs.append(torch.cat(layer_outputs, dim=1))
        
        return outputs[-1]  # (B, T, hidden_size)


class StandardTransformerEncoder(nn.Module):
    """
    Standard Transformer encoder as described in EmMixformer paper (Section II-B-a).
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
            batch_first=False,
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
    Fourier Transformer as described in EmMixformer paper (Section II-B-c).
    Performs self-attention in the frequency domain for global feature learning.
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
        X_f = torch.fft.rfft(x, dim=1, norm='ortho')  # (B, F, C)
        
        # Extract amplitude and phase
        amplitude = torch.abs(X_f)  # (B, F, C)
        phase = torch.angle(X_f)  # (B, F, C)

        # Process with transformers
        amp_out = self.amp_transformer(amplitude)  # (B, F, C)
        phase_out = self.phase_transformer(phase)  # (B, F, C)

        # Recombine
        complex_spec = torch.polar(amp_out, phase_out)  # (B, F, C)

        # Inverse FFT
        x_time = torch.fft.irfft(complex_spec, n=T, dim=1, norm='ortho')  # (B, T, C)
        return x_time


class MixBlock(nn.Module):
    """
    Mix Block as described in EmMixformer paper (Section II-B).
    Combines Attention LSTM, Transformer, and Fourier Transformer.
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

        # Project input to each module's dimension
        self.proj_att = nn.Linear(in_channels, att_hidden_dim)
        self.proj_trans = nn.Linear(in_channels, trans_dim)
        self.proj_fourier = nn.Linear(in_channels, fourier_dim)

        self.att_lstm = AttentionLSTM(att_hidden_dim, att_hidden_dim, num_layers=1)
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
        
        # Initialize projections
        for m in [self.proj_att, self.proj_trans, self.proj_fourier]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        B, C, T = x.shape
        x_t = x.transpose(1, 2)  # (B, T, C)

        # Three parallel branches
        att_in = self.proj_att(x_t)  # (B, T, att_hidden_dim)
        att_out = self.att_lstm(att_in)  # (B, T, att_hidden_dim)

        trans_in = self.proj_trans(x_t)  # (B, T, trans_dim)
        trans_out = self.transformer(trans_in)  # (B, T, trans_dim)

        fourier_in = self.proj_fourier(x_t)  # (B, T, fourier_dim)
        fourier_out = self.fourier_transformer(fourier_in)  # (B, T, fourier_dim)

        # Concatenate features
        mixed = torch.cat([att_out, trans_out, fourier_out], dim=-1)  # (B, T, C_out)
        mixed = mixed.transpose(1, 2)  # (B, C_out, T)
        return mixed


class EmMixformer(nn.Module):
    """
    Full EmMixformer model as described in the paper.
    Architecture: PreprocessingBlock -> SiameseCNN -> MixBlock(s) -> Classifier
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
        num_mix_blocks: int = 2,  # Paper uses 2 mix blocks
    ):
        super().__init__()

        self.preprocess = PreprocessingBlock(velocity_threshold=velocity_threshold)
        self.siamese_cnn = SiameseCNN(
            in_channels=2,
            base_channels=cnn_base_channels,
        )

        # First mix block
        mix_in_channels = self.siamese_cnn.out_channels * 2
        
        # Create mix blocks (paper mentions stacking 2 layers)
        self.mix_blocks = nn.ModuleList()
        for i in range(num_mix_blocks):
            in_ch = mix_in_channels if i == 0 else (mix_att_hidden_dim + mix_trans_dim + mix_fourier_dim)
            self.mix_blocks.append(
                MixBlock(
                    in_channels=in_ch,
                    att_hidden_dim=mix_att_hidden_dim,
                    trans_dim=mix_trans_dim,
                    fourier_dim=mix_fourier_dim,
                    nhead=transformer_heads,
                    num_layers=transformer_layers,
                    dim_feedforward=transformer_ff_dim,
                    dropout=dropout,
                )
            )

        self.feature_dim = mix_att_hidden_dim + mix_trans_dim + mix_fourier_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw eye-movement coordinates, shape (B, T, 2) or (B, 2, T)

        Returns:
            logits: (B, num_classes)
        """
        # Clean input
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Preprocessing
        slow, fast = self.preprocess(x)  # (B, 2, T), (B, 2, T)
        slow = torch.nan_to_num(slow, nan=0.0, posinf=0.0, neginf=0.0)
        fast = torch.nan_to_num(fast, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Siamese CNN
        cnn_out = self.siamese_cnn(slow, fast)  # (B, C_cnn, T_cnn)
        cnn_out = torch.nan_to_num(cnn_out, nan=0.0, posinf=0.0, neginf=0.0)

        # Mix blocks
        mixed = cnn_out
        for mix_block in self.mix_blocks:
            mixed = mix_block(mixed)
            mixed = torch.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)

        # Global average pooling
        feat = mixed.mean(dim=-1)  # (B, C_mix)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Classification
        logits = self.classifier(feat)  # (B, num_classes)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        return logits


if __name__ == "__main__":
    # Sanity check
    batch_size = 4
    seq_len = 512
    num_classes = 10

    model = EmMixformer(num_classes=num_classes)
    dummy_input = torch.randn(batch_size, seq_len, 2)
    out = model(dummy_input)
    print("Output shape:", out.shape)
    print("Model created successfully!")
