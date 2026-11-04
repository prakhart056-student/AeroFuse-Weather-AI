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
        x = x_local.transpose(1, 2)
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.dropout(F.gelu(self.bn2(self.conv2(h))) + h)
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
        last_step = out[:, -1, :]
        return self.head(last_step)


# =====================================================================
# 4. TIER 5 (2026 SOTA): iTransformer (Inverted Multivariate Transformer)
# =====================================================================
class iTransformer2026(nn.Module):
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
        x_inv = x_local.transpose(1, 2)
        tokens = self.variate_embedding(x_inv)
        attended = self.encoder(tokens)
        return self.proj(attended.flatten(start_dim=1))


# =====================================================================
# 5. TIER 5 (2026 SOTA): PatchTST (Patch Time-Series Transformer)
# =====================================================================
class PatchTST2026(nn.Module):
    """Sub-series patching Transformer that groups adjacent timesteps into local patches."""
    def __init__(self, num_features: int, seq_len: int = 24, horizon: int = 24, patch_len: int = 4, stride: int = 2, d_model: int = 64):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        num_patches = (seq_len - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len * num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=0.15, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model * num_patches, horizon)

    def forward(self, x_local, x_grid=None):
        # Unfold time dimension into overlapping patches: (B, num_patches, C_l, patch_len)
        patches = x_local.unfold(dimension=1, size=self.patch_len, step=self.stride)
        B, P, C, L = patches.shape
        patches = patches.reshape(B, P, C * L)
        tokens = self.patch_proj(patches)
        out = self.encoder(tokens)
        return self.head(out.flatten(start_dim=1))


# =====================================================================
# 6. TIER 4 (PROPOSED PAPER MODEL): AeroFuse-Net (v2)
#    Physics-Informed Dual-Modal 2D-CNN + Residual 1D-CNN-BiLSTM +
#    Cross-Modal Gated Attention + Non-Crossing Quantile Uncertainty
# =====================================================================
class SqueezeExcitation2D(nn.Module):
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

        # --- Branch 1: Synoptic 2D-CNN Spatial Grid Encoder ---
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

        # --- Branch 2: Multi-Scale 1D-CNN + Residual BiLSTM Local Encoder ---
        self.local_conv1d = nn.Sequential(
            nn.Conv1d(local_features, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        self.local_bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.local_norm = nn.LayerNorm(d_model)

        # --- Inverted Variate Context Encoder (Captures cross-variable physics) ---
        self.variate_proj = nn.Linear(seq_len, d_model)
        self.variate_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, dropout=0.1, batch_first=True)
        self.variate_to_time = nn.Linear(local_features * d_model, seq_len * d_model)

        # --- Cross-Modal Attention Bridge (Local Queries Synoptic 2D Grid) ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, dropout=0.1, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)

        # --- Learnable Gated Modal Fusion ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # --- Autoregressive Linear Highway + Non-Linear Spatiotemporal Head ---
        self.ar_highway = nn.Linear(seq_len * local_features, horizon)
        self.point_head = nn.Sequential(
            nn.Linear(d_model * seq_len, 128),
            nn.GELU(),
            nn.Linear(128, horizon)
        )

        # --- Decoupled Monotonic Non-Crossing Quantile Offset Heads ---
        self.lower_offset_head = nn.Sequential(
            nn.Linear(d_model * seq_len, 64),
            nn.GELU(),
            nn.Linear(64, horizon)
        )
        self.upper_offset_head = nn.Sequential(
            nn.Linear(d_model * seq_len, 64),
            nn.GELU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x_local, x_grid, return_uncertainty: bool = False):
        B, T, C_g, H, W = x_grid.shape

        # 1. Encode 2D Spatial Grid across all T timesteps
        x_grid_flat = x_grid.view(B * T, C_g, H, W)
        sp_feat = self.se_block(self.spatial_conv(x_grid_flat))
        sp_seq = self.spatial_pool(sp_feat).view(B, T, self.d_model)

        # 2. Residual 1D-CNN + BiLSTM Local Temporal Representation
        loc_cnn = self.local_conv1d(x_local.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.local_bilstm(loc_cnn)
        loc_seq = self.local_norm(loc_cnn + lstm_out)

        # 3. Cross-Variable Physics Context
        var_tokens = self.variate_proj(x_local.transpose(1, 2))
        var_att, _ = self.variate_attn(var_tokens, var_tokens, var_tokens)
        var_context = self.variate_to_time(var_att.flatten(start_dim=1)).view(B, T, self.d_model)
        loc_seq = loc_seq + 0.3 * var_context

        # 4. Cross-Modal Attention: Local State (Q) attends to Synoptic 2D Grid (K, V)
        attn_out, _ = self.cross_attn(query=loc_seq, key=sp_seq, value=sp_seq)
        cross_seq = self.norm_cross(loc_seq + attn_out)

        # 5. Gated Modal Fusion
        gate = self.fusion_gate(torch.cat([loc_seq, cross_seq], dim=-1))
        fused = gate * cross_seq + (1.0 - gate) * loc_seq

        flat = fused.flatten(start_dim=1)

        # 6. Combine Linear Autoregressive Highway + Spatiotemporal Non-Linear Correction
        pred_mean = self.ar_highway(x_local.flatten(start_dim=1)) + self.point_head(flat)

        if return_uncertainty:
            # Stop-gradient decouples quantile training from point forecast accuracy
            # Softplus guarantees q10 <= pred_mean <= q90 (zero quantile crossing!)
            flat_detached = flat.detach()
            base_detached = pred_mean.detach()
            q10 = base_detached - F.softplus(self.lower_offset_head(flat_detached))
            q90 = base_detached + F.softplus(self.upper_offset_head(flat_detached))
            return pred_mean, q10, q90

        return pred_mean


# =====================================================================
# 7. Physics-Informed Smoothness & Non-Crossing Quantile Loss
# =====================================================================
class PhysicsInformedWeatherLoss(nn.Module):
    def __init__(self, lambda_phys: float = 0.10, lambda_quant: float = 0.20):
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

        # First-Order Temporal Derivative Matching (Thermodynamic Continuity Constraint)
        if pred_mean.shape[1] > 1:
            pred_diff = pred_mean[:, 1:] - pred_mean[:, :-1]
            true_diff = y_true[:, 1:] - y_true[:, :-1]
            phys_loss = F.l1_loss(pred_diff, true_diff)
        else:
            phys_loss = 0.0

        total_loss = base_loss + self.lambda_phys * phys_loss

        if q10 is not None and q90 is not None:
            q_loss = self.pinball_loss(y_true, q10, 0.10) + self.pinball_loss(y_true, q90, 0.90)
            total_loss = total_loss + self.lambda_quant * q_loss

        return total_loss