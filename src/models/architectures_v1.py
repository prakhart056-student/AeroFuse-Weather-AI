import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. TIER 3: 1D-CNN Temporal Feature Extractor (Local Station Only)
# =====================================================================
class TemporalCNN1D(nn.Module):
    def __init__(self, num_features: int, seq_len: int = 24, horizon: int = 24, d_model: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(d_model * seq_len, horizon)

    def forward(self, x_local, x_grid=None):
        # x_local: (B, T, C_l) -> transpose to (B, C_l, T) for 1D-CNN
        x = x_local.transpose(1, 2)
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.dropout(F.gelu(self.bn2(self.conv2(h))) + h)  # Residual skip connection
        h = h.flatten(start_dim=1)
        return self.head(h)


# =====================================================================
# 2. TIER 3: 2D-CNN Spatial ResNet (5x5 Synoptic Grid Only)
# =====================================================================
class SpatialResNet2D(nn.Module):
    def __init__(self, grid_channels: int = 6, seq_len: int = 24, horizon: int = 24, d_model: int = 64):
        super().__init__()
        in_ch = grid_channels * seq_len
        self.conv1 = nn.Conv2d(in_ch, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(d_model)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.head = nn.Linear(d_model * 4, horizon)

    def forward(self, x_local, x_grid):
        # x_grid: (B, T, C_g, H=5, W=5) -> collapse (T * C_g) into spatial channels
        B, T, C, H, W = x_grid.shape
        x = x_grid.view(B, T * C, H, W)
        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h))) + h
        h = self.pool(h).flatten(start_dim=1)
        return self.head(h)


# =====================================================================
# 3. TIER 3: Bidirectional LSTM (Sequential Baseline)
# =====================================================================
class BiLSTMForecaster(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 64, horizon: int = 24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15
        )
        self.head = nn.Linear(hidden_dim * 2, horizon)

    def forward(self, x_local, x_grid=None):
        out, _ = self.lstm(x_local)
        last_step = out[:, -1, :]  # (B, hidden_dim * 2)
        return self.head(last_step)


# =====================================================================
# 4. TIER 5 (2026 SOTA): iTransformer (Inverted Multivariate Transformer)
# =====================================================================
class iTransformer2026(nn.Module):
    """
    Inverts the dimensions: embeds the entire temporal lookback of each meteorological
    variable as a variate token, applying Multi-Head Self-Attention across variables.
    """
    def __init__(self, num_features: int, seq_len: int = 24, horizon: int = 24, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.variate_embedding = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.15, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj = nn.Linear(d_model * num_features, horizon)

    def forward(self, x_local, x_grid=None):
        # x_local: (B, T, N) -> invert to (B, N, T)
        x_inv = x_local.transpose(1, 2)
        tokens = self.variate_embedding(x_inv)  # (B, N, d_model)
        attended = self.encoder(tokens)         # (B, N, d_model)
        return self.proj(attended.flatten(start_dim=1))


# =====================================================================
# 5. TIER 4 (PROPOSED PAPER MODEL): AeroFuse-Net
#    Dual-Modal 2D-CNN + 1D-CNN-BiLSTM + Cross-Modal Gated Attention
# =====================================================================
class SqueezeExcitation2D(nn.Module):
    """Channel attention for weighing synoptic atmospheric variables (Pressure vs Wind vs Temp)."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(4, channels // reduction)),
            nn.GELU(),
            nn.Linear(max(4, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class AeroFuseNet(nn.Module):
    def __init__(
        self,
        local_features: int,
        grid_channels: int = 6,
        seq_len: int = 24,
        horizon: int = 24,
        d_model: int = 64
    ):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model

        # --- Branch 1: Synoptic 2D-CNN Spatial Encoder (per timestep) ---
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.se_block = SqueezeExcitation2D(d_model)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- Branch 2: Local 1D-CNN + BiLSTM Microclimate Encoder ---
        self.local_conv1d = nn.Sequential(
            nn.Conv1d(local_features, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        self.local_bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15
        )

        # --- Cross-Modal Attention Bridge (Local Queries Spatial Grid) ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # --- Learnable Gated Modal Fusion ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # --- Multi-Horizon Prediction Head (Point + 10th/90th Quantiles) ---
        self.point_head = nn.Sequential(
            nn.Linear(d_model * seq_len, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, horizon)
        )
        self.lower_quantile_head = nn.Linear(d_model * seq_len, horizon)  # 10th percentile
        self.upper_quantile_head = nn.Linear(d_model * seq_len, horizon)  # 90th percentile

    def forward(self, x_local, x_grid, return_uncertainty: bool = False):
        B, T, C_g, H, W = x_grid.shape

        # 1. Encode 2D Spatial Grid across all T timesteps
        x_grid_flat = x_grid.view(B * T, C_g, H, W)
        sp_feat = self.spatial_conv(x_grid_flat)
        sp_feat = self.se_block(sp_feat)
        sp_seq = self.spatial_pool(sp_feat).view(B, T, self.d_model)  # (B, T, d_model)

        # 2. Encode Local Station Time-Series via 1D-CNN + BiLSTM
        loc_cnn = self.local_conv1d(x_local.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)
        loc_seq, _ = self.local_bilstm(loc_cnn)                               # (B, T, d_model)

        # 3. Cross-Modal Attention: Local State (Q) attends to Synoptic Grid (K, V)
        attn_out, _ = self.cross_attn(query=loc_seq, key=sp_seq, value=sp_seq)
        cross_seq = self.norm1(loc_seq + attn_out)

        # 4. Gated Modal Fusion
        gate = self.fusion_gate(torch.cat([loc_seq, cross_seq], dim=-1))
        fused = gate * cross_seq + (1.0 - gate) * loc_seq  # (B, T, d_model)

        flat = fused.flatten(start_dim=1)
        pred_mean = self.point_head(flat)

        if return_uncertainty:
            q10 = self.lower_quantile_head(flat)
            q90 = self.upper_quantile_head(flat)
            return pred_mean, q10, q90
        return pred_mean


# =====================================================================
# 6. Physics-Informed Smoothness & Quantile Loss Function
# =====================================================================
class PhysicsInformedWeatherLoss(nn.Module):
    """
    Combines Huber (Smooth L1) accuracy loss, temporal gradient continuity penalty
    (preventing non-physical 1-hour temperature shocks), and Pinball Quantile loss.
    """
    def __init__(self, lambda_phys: float = 0.08, lambda_quant: float = 0.15):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.lambda_phys = lambda_phys
        self.lambda_quant = lambda_quant

    @staticmethod
    def pinball_loss(y_true, y_pred, quantile: float):
        errors = y_true - y_pred
        return torch.max((quantile - 1.0) * errors, quantile * errors).mean()

    def forward(self, pred_mean, y_true, q10=None, q90=None):
        base_loss = self.huber(pred_mean, y_true)

        # Temporal First-Order Derivative Matching (Thermodynamic Rate-of-Change Constraint)
        if pred_mean.shape[1] > 1:
            pred_diff = pred_mean[:, 1:] - pred_mean[:, :-1]
            true_diff = y_true[:, 1:] - y_true[:, :-1]
            phys_loss = F.mse_loss(pred_diff, true_diff)
        else:
            phys_loss = 0.0

        total_loss = base_loss + self.lambda_phys * phys_loss

        if q10 is not None and q90 is not None:
            q_loss = self.pinball_loss(y_true, q10, 0.10) + self.pinball_loss(y_true, q90, 0.90)
            total_loss = total_loss + self.lambda_quant * q_loss

        return total_loss