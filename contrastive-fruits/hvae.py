import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        # InstanceNorm is crucial for Style Transfer as it preserves unique image details
        self.norm = nn.InstanceNorm2d(out_ch, affine=True) 

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class StyleEncoder(nn.Module):
    def __init__(self, in_ch=3, base=32, latent_dim=64):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, base)
        self.conv2 = ConvBlock(base, base * 2)
        self.conv3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(base * 4, latent_dim)
        self.fc_logvar = nn.Linear(base * 4, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv3(x)
        x = self.pool(x).view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, content_ch=3, base=32, latent_spatial=7):
        super().__init__()
        self.latent_spatial = latent_spatial
        self.latent_fc = nn.Linear(latent_dim, base * 4 * self.latent_spatial * self.latent_spatial)
        
        # NEW: A specific convolution to expand the Content features
        # Expands 3 channels -> 32 channels (base)
        self.content_encoder = ConvBlock(content_ch, base) 

        # Adjusted input channels: (base*4 from style) + (base from content)
        self.conv1 = ConvBlock(base * 4 + base, base * 4)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = ConvBlock(base * 4, base * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = ConvBlock(base * 2, base)
        self.conv_final = nn.Conv2d(base, 3, kernel_size=3, padding=1)

    def forward(self, content, z):
        # content: Bx3xHxW, z: Bxlatent_dim
        B, C, H, W = content.shape
        
        # 1. Process Style (Latent)
        x = self.latent_fc(z).view(B, -1, self.latent_spatial, self.latent_spatial)
        x = F.interpolate(x, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        
        # 2. Process Content (New Step)
        # Downsample content to match feature map size
        content_small = F.interpolate(content, size=x.shape[2:], mode='bilinear', align_corners=False)
        # Pass through the new encoder to get richer features (3 -> 32 channels)
        content_feat = self.content_encoder(content_small)
        
        # 3. Concatenate (Now Balanced: 128ch + 32ch)
        x = torch.cat([x, content_feat], dim=1)
        
        x = self.conv1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.conv_final(x)
        
        out = torch.sigmoid(x)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


class HVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = StyleEncoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        # initialize weights for stable training
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.InstanceNorm2d):
            if getattr(m, 'weight', None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def encode(self, style_img):
        mu, logvar = self.encoder(style_img)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, content_img, z):
        return self.decoder(content_img, z)

    def forward(self, content_img, style_img):
        mu, logvar = self.encode(style_img)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(content_img, z)
        return recon, mu, logvar

    @torch.no_grad()
    def transfer_style(self, content_img, style_img):
        # content_img, style_img: Bx3xHxW in [0,1]
        mu, logvar = self.encode(style_img)
        z = mu  # use mean for deterministic transfer
        out = self.decode(content_img, z)
        return out


def load_hvae(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    latent_dim = ckpt.get('latent_dim', 64)
    model = HVAE(latent_dim=latent_dim)
    state = ckpt.get('model_state', ckpt)
    # handle DataParallel 'module.' prefixes
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            new_k = k
            if k.startswith('module.'):
                new_k = k[len('module.'):]
            new_state[new_k] = v
        state = new_state
    model.load_state_dict(state)
    model.to(map_location)
    model.eval()
    return model
