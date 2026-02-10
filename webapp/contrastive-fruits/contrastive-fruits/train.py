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
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = FruitStyleDataset(args.fruit_root, args.style_root, image_size=224, train=True, style_method=args.style_method, hvae_ckpt=args.hvae_ckpt, device=args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device(args.device)
    model = get_backbone(pretrained=False, projection_size=args.projection_size, projection_hidden=args.projection_hidden)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        for batch in pbar:
            v1, v2, vcf, labels = batch
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
