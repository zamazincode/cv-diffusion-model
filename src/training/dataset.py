"""
Low-Light Dataset
=================
Düşük ışık iyileştirmesi için veri yükleme ve augmentation.

Desteklenen Veri Setleri:
- LOL (Low-Light Dataset): https://daooshee.github.io/BMVC2018website/
- LOL-v2: https://github.com/flyywh/CVPR-2020-Semi-Low-Light
- SID (See-in-the-Dark): https://github.com/cchen156/Learning-to-See-in-the-Dark
- SICE: https://github.com/csjcai/SICE
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LowLightDataset(Dataset):
    """
    Low-light / Normal-light görüntü çiftleri için dataset.
    
    Dizin yapısı:
    root/
        low/
            image1.png
            image2.png
            ...
        high/
            image1.png
            image2.png
            ...
    """
    
    def __init__(
        self,
        root: str,
        low_dir: str = "low",
        high_dir: str = "high",
        image_size: int = 256,
        augment: bool = True,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
        paired: bool = True,  # Çift görüntüler mi?
    ):
        """
        Args:
            root: Veri seti kök dizini
            low_dir: Düşük ışık görüntüleri alt dizini
            high_dir: Normal ışık görüntüleri alt dizini
            image_size: Çıktı görüntü boyutu
            augment: Augmentation uygula
            extensions: Desteklenen dosya uzantıları
            paired: Görüntüler eşleştirilmiş mi (aynı isimle)
        """
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.paired = paired
        
        # Klasör varlığını kontrol et
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root directory not found: {self.root}\n"
                f"Please download the dataset and ensure the path is correct.\n"
                f"Expected structure:\n"
                f"  {self.root}/\n"
                f"    {low_dir}/\n"
                f"      image1.png\n"
                f"      ...\n"
                f"    {high_dir}/\n"
                f"      image1.png\n"
                f"      ..."
            )
        
        # Görüntü yollarını bul
        low_path = self.root / low_dir
        high_path = self.root / high_dir
        
        # Alternatif klasör yapılarını dene
        if not low_path.exists():
            # LOL dataset yapısı: our485/low veya eval15/low
            # Eğer root zaten our485/eval15 ise, direkt low/high klasörlerini ara
            possible_low_dirs = [low_dir, "low", "lowlight", "dark"]
            possible_high_dirs = [high_dir, "high", "normal", "bright"]
            
            for alt_low in possible_low_dirs:
                alt_path = self.root / alt_low
                if alt_path.exists():
                    low_path = alt_path
                    low_dir = alt_low
                    break
            
            for alt_high in possible_high_dirs:
                alt_path = self.root / alt_high
                if alt_path.exists():
                    high_path = alt_path
                    high_dir = alt_high
                    break
        
        if not low_path.exists():
            raise FileNotFoundError(
                f"Low-light images directory not found: {low_path}\n"
                f"Tried: {[self.root / d for d in ['low', 'lowlight', 'dark']]}\n"
                f"Current directory structure:\n"
                f"{self._list_directory_structure(self.root)}"
            )
        
        if not high_path.exists():
            raise FileNotFoundError(
                f"High-light images directory not found: {high_path}\n"
                f"Tried: {[self.root / d for d in ['high', 'normal', 'bright']]}\n"
                f"Current directory structure:\n"
                f"{self._list_directory_structure(self.root)}"
            )
        
        # Low-light görüntüleri
        self.low_images = sorted([
            f for f in low_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ])
        
        # High-light görüntüleri
        self.high_images = sorted([
            f for f in high_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ])
        
        if len(self.low_images) == 0:
            raise ValueError(
                f"No images found in {low_path}\n"
                f"Supported extensions: {extensions}"
            )
        
        if len(self.high_images) == 0:
            raise ValueError(
                f"No images found in {high_path}\n"
                f"Supported extensions: {extensions}"
            )
        
        # Eşleştirme kontrolü
        if paired:
            if len(self.low_images) != len(self.high_images):
                print(f"Warning: Image count mismatch: {len(self.low_images)} low vs {len(self.high_images)} high")
                print(f"Using minimum count: {min(len(self.low_images), len(self.high_images))}")
                # Minimum sayıya göre kırp
                min_count = min(len(self.low_images), len(self.high_images))
                self.low_images = self.low_images[:min_count]
                self.high_images = self.high_images[:min_count]
        
        # Augmentation pipeline
        self.transform = self._get_transforms()
    
    def _list_directory_structure(self, path: Path, max_depth: int = 2, current_depth: int = 0) -> str:
        """Klasör yapısını listele (hata mesajları için)"""
        if current_depth >= max_depth:
            return ""
        
        lines = []
        try:
            items = sorted(path.iterdir())
            for item in items[:10]:  # İlk 10 öğe
                if item.is_dir():
                    lines.append(f"  {item.name}/")
                    if current_depth < max_depth - 1:
                        sub_lines = self._list_directory_structure(item, max_depth, current_depth + 1)
                        for sub_line in sub_lines.split('\n'):
                            if sub_line.strip():
                                lines.append(f"    {sub_line}")
                else:
                    lines.append(f"  {item.name}")
            if len(items) > 10:
                lines.append(f"  ... ({len(items) - 10} more items)")
        except Exception as e:
            lines.append(f"  (Error listing: {e})")
        
        return '\n'.join(lines)
    
    def _get_transforms(self) -> A.Compose:
        """Augmentation pipeline oluştur"""
        
        transforms = []
        
        if self.augment:
            transforms.extend([
                # Geometric transforms (both images)
                A.RandomCrop(
                    height=self.image_size, 
                    width=self.image_size,
                    p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.3),
            ])
        else:
            transforms.append(
                A.CenterCrop(
                    height=self.image_size,
                    width=self.image_size,
                )
            )
        
        # Normalize to [-1, 1]
        transforms.extend([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
        
        return A.Compose(
            transforms,
            additional_targets={"high": "image"}
        )
    
    def __len__(self) -> int:
        return len(self.low_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Görüntüleri yükle
        low_img = np.array(Image.open(self.low_images[idx]).convert("RGB"))
        
        if self.paired:
            high_img = np.array(Image.open(self.high_images[idx]).convert("RGB"))
        else:
            # Unpaired: random high image
            high_idx = np.random.randint(len(self.high_images))
            high_img = np.array(Image.open(self.high_images[high_idx]).convert("RGB"))
        
        # Augmentation (her iki görüntüye aynı transform)
        transformed = self.transform(image=low_img, high=high_img)
        
        return {
            "low_light": transformed["image"],
            "normal_light": transformed["high"],
            "filename": self.low_images[idx].name,
        }


class SyntheticLowLightDataset(Dataset):
    """
    Sentetik düşük ışık veri seti.
    
    Normal ışıklı görüntülerden sentetik olarak düşük ışıklı
    görüntüler oluşturur. Daha fazla eğitim verisi için kullanılabilir.
    """
    
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        gamma_range: Tuple[float, float] = (2.0, 5.0),
        noise_level_range: Tuple[float, float] = (0.01, 0.05),
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        """
        Args:
            root: Normal ışıklı görüntülerin dizini
            gamma_range: Gamma correction aralığı (karartma için)
            noise_level_range: Gaussian noise seviyesi
        """
        self.root = Path(root)
        self.image_size = image_size
        self.gamma_range = gamma_range
        self.noise_level_range = noise_level_range
        
        self.images = sorted([
            f for f in self.root.iterdir()
            if f.suffix.lower() in extensions
        ])
        
        self.base_transform = A.Compose([
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
        ])
        
        self.to_tensor = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
    
    def _create_low_light(self, image: np.ndarray) -> np.ndarray:
        """Sentetik düşük ışık görüntüsü oluştur"""
        
        # Float'a çevir
        img_float = image.astype(np.float32) / 255.0
        
        # Random gamma correction (karartma)
        gamma = np.random.uniform(*self.gamma_range)
        darkened = np.power(img_float, gamma)
        
        # Gaussian noise ekle
        noise_level = np.random.uniform(*self.noise_level_range)
        noise = np.random.normal(0, noise_level, darkened.shape)
        noisy = np.clip(darkened + noise, 0, 1)
        
        # Color shift (düşük ışıkta renk sapması)
        if np.random.random() < 0.5:
            # Random channel scaling
            scale = np.random.uniform(0.8, 1.0, size=3)
            noisy = noisy * scale
            noisy = np.clip(noisy, 0, 1)
        
        return (noisy * 255).astype(np.uint8)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Normal ışıklı görüntüyü yükle
        normal_img = np.array(Image.open(self.images[idx]).convert("RGB"))
        
        # Base transform
        transformed = self.base_transform(image=normal_img)
        normal_img = transformed["image"]
        
        # Sentetik düşük ışık oluştur
        low_light_img = self._create_low_light(normal_img)
        
        # Tensor'a çevir
        normal_tensor = self.to_tensor(image=normal_img)["image"]
        low_tensor = self.to_tensor(image=low_light_img)["image"]
        
        return {
            "low_light": low_tensor,
            "normal_light": normal_tensor,
            "filename": self.images[idx].name,
        }


def create_dataloaders(
    train_root: str,
    val_root: Optional[str] = None,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_synthetic: bool = False,
    **dataset_kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Training ve validation dataloaders oluştur.
    
    Args:
        train_root: Training veri seti dizini
        val_root: Validation veri seti dizini (None = no validation)
        batch_size: Batch boyutu
        image_size: Görüntü boyutu
        num_workers: DataLoader worker sayısı
        use_synthetic: Sentetik veri seti kullan
    """
    
    if use_synthetic:
        train_dataset = SyntheticLowLightDataset(
            root=train_root,
            image_size=image_size,
            **dataset_kwargs,
        )
    else:
        train_dataset = LowLightDataset(
            root=train_root,
            image_size=image_size,
            augment=True,
            **dataset_kwargs,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = None
    if val_root is not None:
        val_dataset = LowLightDataset(
            root=val_root,
            image_size=image_size,
            augment=False,
            **dataset_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test synthetic dataset
    import tempfile
    import os
    
    # Fake data oluştur
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test görüntüleri oluştur
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            img.save(os.path.join(tmpdir, f"test_{i}.png"))
        
        # Dataset test
        dataset = SyntheticLowLightDataset(tmpdir, image_size=256)
        sample = dataset[0]
        
        print(f"Low-light shape: {sample['low_light'].shape}")
        print(f"Normal-light shape: {sample['normal_light'].shape}")
        print(f"Low-light range: [{sample['low_light'].min():.2f}, {sample['low_light'].max():.2f}]")
        print(f"Normal-light range: [{sample['normal_light'].min():.2f}, {sample['normal_light'].max():.2f}]")

