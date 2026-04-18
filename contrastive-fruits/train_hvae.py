import argparse
import os
import random
import torch
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from utils import find_images
from hvae import HVAE
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
from torchvision.utils import save_image

# --- DATASET CLASSES ---
class StylePairDataset(Dataset):
    def __init__(self, content_root, style_root, image_size=224):
        self.content_paths = find_images(content_root)
        self.style_paths = find_images(style_root)
        self.image_size = image_size
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        if len(self.content_paths) == 0 or len(self.style_paths) == 0:
            raise RuntimeError('No images found for HVAE training.')
    def __len__(self):
        return len(self.content_paths) * len(self.style_paths)
    def __getitem__(self, idx):
        num_styles = len(self.style_paths)
        content_idx = idx // num_styles
        style_idx = idx % num_styles
        c = Image.open(self.content_paths[content_idx]).convert('RGB')
        s = Image.open(self.style_paths[style_idx]).convert('RGB')
        return self.transform(c), self.transform(s)

class StyleAllPairsDataset(Dataset):
    def __init__(self, content_root, style_root, image_size=224):
        self.content_paths = find_images(content_root)
        self.style_paths = find_images(style_root)
        self.image_size = image_size
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        if len(self.content_paths) == 0 or len(self.style_paths) == 0:
            raise RuntimeError('No images found for HVAE training.')
        self.num_contents = len(self.content_paths)
        self.num_styles = len(self.style_paths)
    def __len__(self):
        return self.num_contents * self.num_styles
    def __getitem__(self, idx):
        content_idx = idx // self.num_styles
        style_idx = idx % self.num_styles
        c = Image.open(self.content_paths[content_idx]).convert('RGB')
        s = Image.open(self.style_paths[style_idx]).convert('RGB')
        return self.transform(c), self.transform(s), content_idx

# --- NEW: VGG Feature Extractor (This is the critical fix) ---
class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except:
            vgg = models.vgg16(pretrained=True).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        # Slice 1 (Layers 0-4): Catches COLORS
        for x in range(4): self.slice1.add_module(str(x), vgg[x])
        # Slice 2 (Layers 4-9): Catches TEXTURES
        for x in range(4, 9): self.slice2.add_module(str(x), vgg[x])
        # Slice 3 (Layers 9-16): Catches PATTERNS
        for x in range(9, 16): self.slice3.add_module(str(x), vgg[x])
        # Slice 4 (Layers 16-23): Catches STRUCTURE
        for x in range(16, 23): self.slice4.add_module(str(x), vgg[x])
        
        self.to(device)
        self.eval()
        for p in self.parameters(): p.requires_grad = False
        
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]

def gram_matrix(input):
    B, C, H, W = input.size()
    features = input.view(B, C, H * W)
    G = torch.bmm(features, features.transpose(1, 2)) 
    return G.div(C * H * W)

def group_collate(batch):
    groups = OrderedDict()
    for item in batch:
        # item may be (c, s, ci) for StyleAllPairsDataset
        if len(item) == 3:
            c, s, ci = item
        else:
            # fallback
            c, s = item
            ci = 0
        if ci not in groups:
            groups[ci] = {'content': c, 'styles': []}
        groups[ci]['styles'].append(s)
    contents = []
    styles = []
    for v in groups.values():
        contents.append(v['content'])
        styles.append(torch.stack(v['styles'], dim=0))
    contents = torch.stack(contents, dim=0)
    Ks = [s.shape[0] for s in styles]
    if len(set(Ks)) == 1:
        styles = torch.stack(styles, dim=0)
    else:
        max_k = max(Ks)
        padded = []
        for s in styles:
            k = s.shape[0]
            if k < max_k:
                pad = s[-1].unsqueeze(0).repeat(max_k - k, 1, 1, 1)
                s = torch.cat([s, pad], dim=0)
            padded.append(s)
        styles = torch.stack(padded, dim=0)
    return contents, styles
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--content-root', type=str, required=True)
    p.add_argument('--style-root', type=str, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--save-dir', type=str, default='hvae_checkpoints')
    
    # Weights optimized for Gram Matrices
    p.add_argument('--perc-weight', type=float, default=1e5) 
    p.add_argument('--content-perc-weight', type=float, default=1.0) 
    p.add_argument('--kld-weight', type=float, default=1e-5)
    
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--pair-mode', type=str, default='all', choices=['random', 'all'])
    p.add_argument('--max-workers', type=int, default=4)
    p.add_argument('--samples-per-epoch', type=int, default=0)
    p.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from (e.g., hvae_checkpoints/hvae_epoch_5.pt)')
    return p.parse_args()

def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print('torch.cuda.is_available():', torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print('cuda device count:', torch.cuda.device_count())
            print('cuda device name:', torch.cuda.get_device_name(0))
        except Exception as e:
            print('Could not query CUDA device name:', e)

    # Dataset (Unchanged but with Windows num_workers fix)
    workers = 0 if os.name == 'nt' else args.max_workers
    if args.pair_mode == 'all':
        ds = StyleAllPairsDataset(args.content_root, args.style_root, image_size=224)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=(device.type=='cuda'), collate_fn=group_collate, persistent_workers=False)
    else:
        ds = StylePairDataset(args.content_root, args.style_root, image_size=224)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=(device.type=='cuda'), persistent_workers=False)

    model = HVAE(latent_dim=args.latent_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        import re
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        
        if 'optimizer_state' in ckpt:
            opt.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        
        match = re.search(r'epoch_(\d+)', args.resume)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"Loaded weights. Resuming training at epoch {start_epoch}")
        elif 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            print(f"Loaded weights. Resuming training at epoch {start_epoch}")

        # Fast-forward scheduler only if it wasn't saved in checkpoint
        if 'scheduler_state' not in ckpt:
            for _ in range(start_epoch - 1):
                scheduler.step()

    # Use the new Multi-Layer VGG
    vgg_net = VGGFeatureExtractor(device)
    print("Training with Multi-Layer VGG (Color + Texture)...")
    # Sanity prints: verify model and VGG are on the expected device
    try:
        first_param = next(model.parameters())
        print('model first param device:', first_param.device)
    except StopIteration:
        print('model has no parameters')
    try:
        # that vgg_net has mean/std buffers on device
        print('vgg mean device:', vgg_net.mean.device)
    except Exception:
        pass

    target_batches = len(loader)
    if args.samples_per_epoch > 0:
        target_batches = math.ceil(args.samples_per_epoch / args.batch_size)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        samples_dir = os.path.join(args.save_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        pbar = tqdm(total=target_batches, desc=f"HVAE Epoch {epoch}/{args.epochs}")
        running = 0.0
        batch_count = 0
        
        for batch in loader:
            if args.pair_mode == 'all':
                c_batch, s_batch = batch
                c_batch = c_batch.to(device)
                s_batch = s_batch.to(device)
                B, K = s_batch.shape[0], s_batch.shape[1]
                c_in = c_batch.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(-1, c_batch.size(1), c_batch.size(2), c_batch.size(3))
                s_in = s_batch.reshape(-1, s_batch.size(2), s_batch.size(3), s_batch.size(4))
            else:
                c, s = batch
                c_in = c.to(device)
                s_in = s.to(device)

            recon, mu, logvar = model(c_in, s_in)

            # Extract features from 4 layers
            recon_feats = vgg_net(recon)
            with torch.no_grad():
                style_feats = vgg_net(s_in)
                content_feats = vgg_net(c_in)

            # Style Loss: Sum of Gram Matrices for ALL 4 layers
            loss_style = 0.0
            for rf, sf in zip(recon_feats, style_feats):
                loss_style += F.mse_loss(gram_matrix(rf), gram_matrix(sf))
            
            # Content Loss: MSE on just the deep structure layer (Layer 3)
            loss_content = F.mse_loss(recon_feats[3], content_feats[3])

            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = (args.perc_weight * loss_style) + \
                   (args.content_perc_weight * loss_content) + \
                   (args.kld_weight * kld)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            batch_count += 1
            pbar.update(1)
            pbar.set_postfix({'loss': running / batch_count})

            if batch_count == 1:
                # Print runtime info on the first processed batch to confirm allocations
                print('--- runtime check (first batch) ---')
                try:
                    print('c_in device:', c_in.device)
                except Exception:
                    pass
                try:
                    print('recon device:', recon.device)
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        print('cuda memory allocated:', torch.cuda.memory_allocated())
                        print('cuda memory reserved :', torch.cuda.memory_reserved())
                    except Exception as e:
                        print('cuda memory query failed:', e)
                with torch.no_grad():
                    n = min(4, c_in.size(0))
                    grid = torch.cat([c_in[:n], s_in[:n], recon[:n]], dim=0)
                    save_image(grid, os.path.join(samples_dir, f'epoch_{epoch:03d}_samples.png'), nrow=n)

            if batch_count >= target_batches:
                break
        
        scheduler.step()
        ckpt = {
            'model_state': model.state_dict(),
            'latent_dim': args.latent_dim,
            'optimizer_state': opt.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'hvae_epoch_{epoch}.pt'))

    print('HVAE training finished.')

if __name__ == '__main__':
    train()