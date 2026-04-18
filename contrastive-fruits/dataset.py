import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
from utils import match_channel_stats, find_images, ensure_pil, reinhard_color_transfer, numpy_to_pil, apply_masked_color_transfer


_HAVE_HVAE = False
try:
    # lazy import to avoid hard dependency when HVAE isn't used
    from hvae import load_hvae
    _HAVE_HVAE = True
except Exception:
    _HAVE_HVAE = False


class FruitStyleDataset(Dataset):
    """Dataset that returns: (view1_tensor, view2_tensor, counterfactual_tensor, label)
    - view1/view2: two standard augmentations (positives)
    - counterfactual: style-transferred view using a random phone-style image
    - label: optional class label (if folder structure contains class names)
    """

    def __init__(self, fruit_root, style_root, image_size=224, train=True, style_method='simple', hvae_ckpt=None, device='cpu', feather_radius: int = 8):
        self.fruit_paths = find_images(fruit_root)
        self.style_paths = find_images(style_root)
        if len(self.fruit_paths) == 0:
            raise RuntimeError(f"No fruit images found under {fruit_root}")
        if len(self.style_paths) == 0:
            raise RuntimeError(f"No style images found under {style_root}")

        self.train = train
        self.image_size = image_size
        self.style_method = style_method
        self.hvae_ckpt = hvae_ckpt
        self.device = device
        self.feather_radius = feather_radius

        # ensure CUDA availability if requested
        if isinstance(self.device, str) and self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {self.device} but CUDA is not available. Aborting to avoid accidental CPU run.")

        # if using HVAE, try to load it
        self.hvae = None
        if self.style_method == 'hvae':
            if not _HAVE_HVAE:
                raise RuntimeError("HVAE support requested but 'hvae.py' could not be imported.")
            if self.hvae_ckpt is None:
                raise RuntimeError("style_method='hvae' requires a valid --hvae-ckpt path")
            self.hvae = load_hvae(self.hvae_ckpt, map_location=self.device)
            self.hvae.eval()

        # augmentations similar to contrastive learning
        self.augment = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augment_no_norm = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3),
        ])

    def __len__(self):
        return len(self.fruit_paths)

    def _load_image(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, idx):
        fruit_path = self.fruit_paths[idx]
        fruit_img = self._load_image(fruit_path)

        # two augmented views
        view1 = self.augment(fruit_img)
        view2 = self.augment(fruit_img)

        # create counterfactual by applying style from a random style image
        style_path = random.choice(self.style_paths)
        style_img = self._load_image(style_path)

        # use a no-normalize augmentation to keep spatial cropping similar before style transfer
        content_for_transfer = self.augment_no_norm(fruit_img)
        # content_for_transfer is a PIL transform result? It may not be PIL if ToTensor used; keep it as PIL by applying on original
        # Instead, use a simple resize + crop for style transfer
        content_small = fruit_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        style_small = style_img.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.style_method == 'hvae':
            # generate counterfactual using the pretrained HVAE model
            # convert PIL to tensor in [0,1]
            content_tensor = T.ToTensor()(content_small).unsqueeze(0).to(self.device)
            style_tensor = T.ToTensor()(style_small).unsqueeze(0).to(self.device)
            with torch.no_grad():
                gen = self.hvae.transfer_style(content_tensor, style_tensor)
            # gen is a tensor in [0,1], convert to PIL so existing augment pipeline can be reused
            gen_np = gen.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            counterfactual_pil = numpy_to_pil(gen_np)
            counterfactual = self.augment(counterfactual_pil)
        else:
            # if content has alpha / mask, extract it and use masked color transfer with feathering
            mask = None
            if getattr(fruit_img, 'mode', None) == 'RGBA':
                # extract alpha channel resized to image_size
                alpha = fruit_img.split()[-1]
                mask = alpha.resize((self.image_size, self.image_size), Image.BILINEAR)

            if self.style_method == 'reinhard':
                counterfactual_pil = apply_masked_color_transfer(content_small, style_small, mask=mask, method='reinhard', feather_radius=self.feather_radius)
            else:
                counterfactual_pil = apply_masked_color_transfer(content_small, style_small, mask=mask, method='match', feather_radius=self.feather_radius)
            # apply final augmentations and normalization
            counterfactual = self.augment(counterfactual_pil)

        # attempt to infer label from directory name (folder name directly under fruit_root)
        label = -1
        try:
            parts = os.path.normpath(fruit_path).split(os.sep)
            # assume class is immediate folder containing the image or first folder under fruit_root
            label = parts[-2]
        except Exception:
            label = -1

        return view1, view2, counterfactual, label
