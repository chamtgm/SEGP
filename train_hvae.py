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
from torchvision.utils import save_image


class StylePairDataset(Dataset):
    """Yields (content_img_tensor, style_img_tensor) pairs for HVAE training.
    Images are returned as tensors in [0,1]."""

    def __init__(self, content_root, style_root, image_size=224):
        self.content_paths = find_images(content_root)
        self.style_paths = find_images(style_root)
        self.image_size = image_size
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

        if len(self.content_paths) == 0 or len(self.style_paths) == 0:
            raise RuntimeError('No images found for HVAE training.')

    def __len__(self):
        # iterate over every content x style pair
        return len(self.content_paths) * len(self.style_paths)

    def __getitem__(self, idx):
        # map linear idx to (content_idx, style_idx) so each style is paired
        # with every content deterministically
        num_styles = len(self.style_paths)
        content_idx = idx // num_styles
        style_idx = idx % num_styles

        c_path = self.content_paths[content_idx]
        s_path = self.style_paths[style_idx]
        c = Image.open(c_path).convert('RGB')
        s = Image.open(s_path).convert('RGB')
        return self.transform(c), self.transform(s)


class StyleAllPairsDataset(Dataset):
    """Yields every (content, style) pair. Useful when you want each anchor mapped to all counterfactuals.
    The dataset length is len(content_paths) * len(style_paths)."""

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
        # map linear idx to (content_idx, style_idx)
        content_idx = idx // self.num_styles
        style_idx = idx % self.num_styles

        c_path = self.content_paths[content_idx]
        s_path = self.style_paths[style_idx]

        c = Image.open(c_path).convert('RGB')
        s = Image.open(s_path).convert('RGB')
        # return content, style, and content_idx to allow grouping in collate_fn
        return self.transform(c), self.transform(s), content_idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--content-root', type=str, required=True, help='Path to content images (fruit dataset)')
    p.add_argument('--style-root', type=str, required=True, help='Path to style images (phone-style dataset)')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--save-dir', type=str, default='hvae_checkpoints')
    p.add_argument('--l1-weight', type=float, default=1.0, help='Weight for L1 reconstruction loss')
    p.add_argument('--perc-weight', type=float, default=1.0, help='Weight for perceptual (VGG) loss')
    p.add_argument('--kld-weight', type=float, default=1e-4, help='Weight for KL divergence')
    p.add_argument('--content-perc-weight', type=float, default=1.0, help='Weight for content perceptual loss (preserve structure)')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--pair-mode', type=str, default='all', choices=['random', 'all'],
                   help='How to pair content and style images: "random" samples a random style per content, "all" yields all content×style pairs')
    p.add_argument('--max-workers', type=int, default=4, help='Number of DataLoader workers')
    p.add_argument('--samples-per-epoch', type=int, default=0,
                   help='Limit number of samples (content×style pairs) processed per epoch; 0 means full dataset')
    return p.parse_args()


def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # determine device early so DataLoader pin_memory can be set correctly
    device = torch.device(args.device)

    if args.pair_mode == 'all':
        ds = StyleAllPairsDataset(args.content_root, args.style_root, image_size=224)
        # custom collate: group styles per content index inside each batch
        def group_collate(batch):
            # batch: list of (content, style, content_idx)
            from collections import OrderedDict
            groups = OrderedDict()
            for c, s, ci in batch:
                if ci not in groups:
                    groups[ci] = {'content': c, 'styles': []}
                groups[ci]['styles'].append(s)

            contents = []
            styles = []
            for v in groups.values():
                contents.append(v['content'])
                # stack styles for this content: (K, C, H, W)
                styles.append(torch.stack(v['styles'], dim=0))

            # contents: list of (C,H,W) -> stack to (B, C, H, W)
            contents = torch.stack(contents, dim=0)
            # styles: list of (K_i, C, H, W) -> pad to common K (use shortest K in batch or keep ragged as list)
            # For simplicity, require batch to have uniform K (typical when dataset and batch boundaries align). Convert to tensor (B, K, C, H, W).
            Ks = [s.shape[0] for s in styles]
            if len(set(Ks)) == 1:
                styles = torch.stack(styles, dim=0)
            else:
                # pad shorter style lists with the last style to match max K
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

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.max_workers, pin_memory=(device.type=='cuda'), collate_fn=group_collate)
    else:
        ds = StylePairDataset(args.content_root, args.style_root, image_size=224)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.max_workers, pin_memory=(device.type=='cuda'))

    # compute how many batches to run per epoch (allow capping to reduce epoch size)
    dataset_len = len(ds)
    default_batches = math.ceil(dataset_len / args.batch_size)
    if args.samples_per_epoch and args.samples_per_epoch > 0:
        target_batches = min(default_batches, math.ceil(args.samples_per_epoch / args.batch_size))
    else:
        target_batches = default_batches

    # friendly check: avoid triggering low-level CUDA init error when torch was built CPU-only
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (--device cuda) but PyTorch is not CUDA-enabled on this environment.\n"
                           "Detected GPU drivers (nvidia-smi) may be present but the installed torch wheel is CPU-only.\n"
                           "To fix: install a CUDA-enabled PyTorch build (see https://pytorch.org/get-started/locally) or run with --device cpu.")
    model = HVAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # perceptual feature extractor (VGG16 conv features)
    # use modern torchvision weights API when available
    try:
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device)
    except Exception:
        vgg = models.vgg16(pretrained=True).features[:16].to(device)
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    # precompute normalization tensors once
    vgg_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    vgg_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def perceptual_loss(x, y):
        # x, y are tensors in [0,1]
        # normalize to VGG expected range and compute features under no_grad
        x_n = (x - vgg_mean) / vgg_std
        y_n = (y - vgg_mean) / vgg_std
        with torch.no_grad():
            fx = vgg(x_n)
            fy = vgg(y_n)
        return F.mse_loss(fx, fy)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # ensure sample dir exists for qualitative outputs
        samples_dir = os.path.join(args.save_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        # use a capped tqdm with `target_batches` steps to limit GPU work per epoch
        pbar = tqdm(total=target_batches, desc=f"HVAE Epoch {epoch}/{args.epochs}")
        running = 0.0
        batch_count = 0
        for batch in loader:
            if args.pair_mode == 'all':
                # grouped batch: contents (B, C, H, W), styles (B, K, C, H, W)
                c_batch, s_batch = batch
                c_batch = c_batch.to(device)
                s_batch = s_batch.to(device)
                B, K = s_batch.shape[0], s_batch.shape[1]
                # expand contents to match styles and flatten for model input
                c_exp = c_batch.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(-1, c_batch.size(1), c_batch.size(2), c_batch.size(3))
                s_flat = s_batch.reshape(-1, s_batch.size(2), s_batch.size(3), s_batch.size(4))
                recon, mu, logvar = model(c_exp, s_flat)
            # unify variable names for downstream loss code
            if args.pair_mode == 'all':
                # for the "all" mode we expanded contents and flattened styles
                # to feed the model; use those tensors for loss computation so
                # the later code can reference `c` and `s` uniformly.
                s = s_flat
                c = c_exp
            else:
                c, s = batch
                c = c.to(device)
                s = s.to(device)
                recon, mu, logvar = model(c, s)
            # L1 reconstruction (reduces blurring vs MSE)
            recon_l1 = F.l1_loss(recon, s)
            # perceptual loss (VGG features)
            perc = perceptual_loss(recon, s)
            # perceptual content loss: preserve structure of content image
            perc_content = perceptual_loss(recon, c)
            # KLD
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = args.l1_weight * recon_l1 + args.perc_weight * perc + args.content_perc_weight * perc_content + args.kld_weight * kld

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            batch_count += 1
            pbar.update(1)
            pbar.set_postfix({'loss': running / batch_count})

            # save first batch reconstructions for the epoch (only once per epoch)
            if batch_count == 1:
                try:
                    n = 4
                    if args.pair_mode == 'all':
                        # c_batch: (B, C, H, W), s_batch: (B, K, C, H, W), recon: (B*K, C, H, W)
                        B = c_batch.size(0)
                        K = s_batch.size(1)
                        n_show = min(n, B)
                        # pick first content and its first up to 3 styles
                        content_imgs = c_batch[:n_show]
                        styles_imgs = s_batch[:n_show, :min(3, K)]
                        # get reconstructions corresponding to first content block
                        recon_images = recon.reshape(B, K, recon.size(1), recon.size(2), recon.size(3))[:n_show, :min(3, K)]
                        # for display, interleave content, styles, recons for each sample
                        display_list = []
                        for i in range(n_show):
                            display_list.append(content_imgs[i])
                            # append styles
                            for j in range(styles_imgs.shape[1]):
                                display_list.append(styles_imgs[i, j])
                            # append recon for same styles
                            for j in range(recon_images.shape[1]):
                                display_list.append(recon_images[i, j])
                        grid = torch.stack(display_list, dim=0)
                        save_path = os.path.join(samples_dir, f'epoch_{epoch:03d}_samples.png')
                        save_image(grid, save_path, nrow=min(1 + styles_imgs.shape[1] * 2, grid.size(0)))
                    else:
                        # create a grid: content / style / recon stacked vertically per sample
                        n = min(4, c.size(0))
                        grid = torch.cat([c[:n], s[:n], recon[:n]], dim=0)
                        save_path = os.path.join(samples_dir, f'epoch_{epoch:03d}_samples.png')
                        save_image(grid, save_path, nrow=n)
                except Exception as e:
                    print(f"Could not save samples for epoch {epoch}: {e}")

            # stop early if we've reached the target number of batches for this epoch
            if batch_count >= target_batches:
                break

        # save checkpoint
        ckpt = {'model_state': model.state_dict(), 'latent_dim': args.latent_dim}
        torch.save(ckpt, os.path.join(args.save_dir, f'hvae_epoch_{epoch}.pt'))

    print('HVAE training finished.')


if __name__ == '__main__':
    train()
