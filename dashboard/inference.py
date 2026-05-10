import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import json
import os

# ─── Building Blocks ───
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, padding=0, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        attn = self.sigmoid(psi)
        if attn.shape[2:] != x.shape[2:]:
            attn = F.interpolate(attn, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x * attn


# ─── Model Architectures ───
class MetadataUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, meta_features=20):
        super().__init__()
        self.down1 = ResidualBlock(in_channels, 64)
        self.down2 = ResidualBlock(64, 128)
        self.down3 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.meta_dim = 32
        self.meta_mlp = nn.Sequential(nn.Linear(meta_features, 64), nn.ReLU(), nn.Linear(64, self.meta_dim), nn.ReLU())
        self.bottleneck = ResidualBlock(256 + self.meta_dim, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = ResidualBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = ResidualBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = ResidualBlock(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, img, meta):
        x1 = self.down1(img)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        pooled_x3 = self.pool(x3)
        meta_feat = self.meta_mlp(meta)
        meta_expanded = meta_feat.view(meta_feat.size(0), self.meta_dim, 1, 1)
        meta_expanded = meta_expanded.expand(-1, -1, pooled_x3.size(2), pooled_x3.size(3))
        x4 = self.bottleneck(torch.cat([pooled_x3, meta_expanded], dim=1))
        x = self.up_conv1(torch.cat([self.up1(x4), x3], dim=1))
        x = self.up_conv2(torch.cat([self.up2(x), x2], dim=1))
        x = self.up_conv3(torch.cat([self.up3(x), x1], dim=1))
        return self.out_conv(x)


class AttentionMetadataUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, meta_features=20):
        super().__init__()
        self.down1 = ResidualBlock(in_channels, 64)
        self.down2 = ResidualBlock(64, 128)
        self.down3 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.meta_dim = 32
        self.meta_mlp = nn.Sequential(nn.Linear(meta_features, 64), nn.ReLU(), nn.Linear(64, self.meta_dim), nn.ReLU())
        self.bottleneck = ResidualBlock(256 + self.meta_dim, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.up_conv1 = ResidualBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.up_conv2 = ResidualBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.up_conv3 = ResidualBlock(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, img, meta):
        x1 = self.down1(img)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        pooled_x3 = self.pool(x3)
        meta_feat = self.meta_mlp(meta)
        me = meta_feat.view(meta_feat.size(0), self.meta_dim, 1, 1)
        me = me.expand(-1, -1, pooled_x3.size(2), pooled_x3.size(3))
        x4 = self.bottleneck(torch.cat([pooled_x3, me], dim=1))
        d1 = self.up1(x4)
        x3 = self.attn1(d1, x3)
        x = self.up_conv1(torch.cat([d1, x3], dim=1))
        d2 = self.up2(x)
        x2 = self.attn2(d2, x2)
        x = self.up_conv2(torch.cat([d2, x2], dim=1))
        d3 = self.up3(x)
        x1 = self.attn3(d3, x1)
        x = self.up_conv3(torch.cat([d3, x1], dim=1))
        return self.out_conv(x)


# ─── Helpers ───
MODEL_CLASSES = {
    'MetadataUNet': MetadataUNet,
    'AttentionMetadataUNet': AttentionMetadataUNet,
}

def load_model(model_path, device, model_class='MetadataUNet', meta_features=20):
    cls = MODEL_CLASSES.get(model_class, MetadataUNet)
    model = cls(meta_features=meta_features)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_bytes, image_size=(128, 128)):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    original_size = (img.shape[1], img.shape[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0), original_size


def build_metadata_vector(age, gender, histology, location, rna_cluster, methyl_cluster, feature_mapping):
    """Build a 20-dim metadata vector using the EXACT column order from training."""
    columns = feature_mapping['columns']
    vec = np.zeros(len(columns), dtype=np.float32)

    # Age (StandardScaled)
    mean = feature_mapping['scaler_mean']
    scale = feature_mapping['scaler_scale']
    vec[0] = (age - mean) / scale

    # One-hot categoricals via the mapping
    cat_labels = feature_mapping['categorical_labels']

    # Gender
    gender_col = cat_labels['gender'].get(gender)
    if gender_col and gender_col in columns:
        vec[columns.index(gender_col)] = 1.0

    # Histological type
    hist_col = cat_labels['histological_type'].get(histology)
    if hist_col and hist_col in columns:
        vec[columns.index(hist_col)] = 1.0

    # Tumor location
    loc_col = cat_labels['tumor_location'].get(location)
    if loc_col and loc_col in columns:
        vec[columns.index(loc_col)] = 1.0

    # RNA cluster
    rna_col = cat_labels['RNASeqCluster'].get(str(rna_cluster))
    if rna_col and rna_col in columns:
        vec[columns.index(rna_col)] = 1.0

    # Methylation cluster
    meth_col = cat_labels['MethylationCluster'].get(str(methyl_cluster))
    if meth_col and meth_col in columns:
        vec[columns.index(meth_col)] = 1.0

    return vec
