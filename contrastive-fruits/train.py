import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset import FruitStyleDataset
from models import get_backbone
from losses import combined_counterfactual_loss
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fruit-root', type=str, required=True, help='Path to fruit dataset root (e.g., "Dataset, Angle Variable")')
    p.add_argument('--style-root', type=str, required=True, help='Path to phone style images root (e.g., phone-style-output/dataset_extracted)')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--projection-size', type=int, default=128)
    p.add_argument('--projection-hidden', type=int, default=512)
    p.add_argument('--alpha', type=float, default=1.0, help='weight for counterfactual loss term')
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--save-dir', type=str, default='checkpoints')
    p.add_argument('--style-method', type=str, default='simple', choices=['simple', 'reinhard', 'hvae'], help='Style transfer method to create counterfactuals')
    p.add_argument('--hvae-ckpt', type=str, default=None, help='Path to pretrained HVAE checkpoint (required when --style-method=hvae)')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers')
    p.add_argument('--resume', action='store_true', help='Resume from latest checkpoint in --save-dir')
    p.add_argument('--resume-ckpt', type=str, default=None, help='Path to specific checkpoint to resume from (overrides --resume)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = FruitStyleDataset(args.fruit_root, args.style_root, image_size=224, train=True, style_method=args.style_method, hvae_ckpt=args.hvae_ckpt, device=args.device)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0 if os.name == 'nt' else args.num_workers, # Fix Windows multiprocessing issues
        pin_memory=True,
        persistent_workers=False
    )

    device = torch.device(args.device)
    model = get_backbone(pretrained=False, projection_size=args.projection_size, projection_hidden=args.projection_hidden)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume_ckpt:
        ckpt_path = args.resume_ckpt
        if not os.path.exists(ckpt_path):
            alt = os.path.join(args.save_dir, ckpt_path)
            if os.path.exists(alt):
                ckpt_path = alt
        if not os.path.exists(ckpt_path):
            print('Checkpoint not found at', args.resume_ckpt)
            raise SystemExit(1)
        print(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                print('Warning: optimizer state could not be loaded (incompatible). Starting with fresh optimizer state.')
        start_epoch = ckpt.get('epoch', 0) + 1
    elif args.resume:
        # discover latest checkpoint in save dir
        import glob, re
        ckpts = glob.glob(os.path.join(args.save_dir, 'ckpt_epoch_*.pt'))
        if ckpts:
            # extract epoch numbers
            def epoch_from_name(p):
                m = re.search(r'ckpt_epoch_(\d+)\.pt$', p)
                return int(m.group(1)) if m else -1
            ckpts_sorted = sorted(ckpts, key=epoch_from_name)
            latest = ckpts_sorted[-1]
            print(f'Loading checkpoint {latest}')
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            if 'optimizer_state' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                except Exception:
                    print('Warning: optimizer state could not be loaded (incompatible). Starting with fresh optimizer state.')
            start_epoch = ckpt.get('epoch', 0) + 1
        else:
            print('No checkpoint found in', args.save_dir, '; starting from scratch')

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for batch in pbar:
            v1, v2, vcf = batch
            v1 = v1.to(device)
            v2 = v2.to(device)
            vcf = vcf.to(device)

            z1 = model(v1)
            z2 = model(v2)
            zcf = model(vcf)

            loss = combined_counterfactual_loss(z1, z2, zcf, alpha=args.alpha, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        # save checkpoint
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_epoch_{epoch}.pt'))

    print('Training finished.')


if __name__ == '__main__':
    main()
