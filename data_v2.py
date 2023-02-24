import os
from PIL import Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from augment import FRandomCrop, FRandomHorizontalFilp, FToTensor, FCompose


def train_dataloader(image_dir, batch_size=64, num_workers=0, use_transform=True):
    transform = None
    if use_transform:
        transform = None
    dataloader = DataLoader(
        RainDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(image_dir, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        RainDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

class RainDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.image_list = os.listdir(os.path.join(image_dir, 'input/')) 
        self._check_image(self.image_list)
        self.label_list = list()
        for i in range(len(self.image_list)):
            filename = self.image_list[i]
            self.label_list.append(filename)
        self.image_list.sort()
        self.label_list.sort()
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx])).convert("RGB")
        label = Image.open(os.path.join(self.image_dir, 'target', self.label_list[idx])).convert("RGB")
        ldgp = Image.open(os.path.join(self.image_dir, 'ldgp', self.label_list[idx])).convert("L")
        
        if self.transform:
            image, label, ldgp = self.transform(image, label, ldgp)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            ldgp = F.to_tensor(ldgp)
        
        name = self.image_list[idx]
        return image, label, ldgp, name
        
    
    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError