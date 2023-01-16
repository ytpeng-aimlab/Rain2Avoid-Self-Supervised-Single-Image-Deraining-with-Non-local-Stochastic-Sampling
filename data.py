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
        Dataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(image_dir, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        Dataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

class Dataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.image_list = os.listdir(os.path.join(image_dir, 'rainy/')) 
        self._check_image(self.image_list) # 檢查檔案類型
        self.label_list = list()
        for i in range(len(self.image_list)):
            filename = self.image_list[i]
            self.label_list.append(filename)
        self.image_list.sort()
        self.label_list.sort()
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'rainy', self.image_list[idx])).convert("RGB")
        label = Image.open(os.path.join(self.image_dir, 'gt', self.label_list[idx])).convert("RGB")
        # mask = Image.open(os.path.join(self.image_dir, 'mask', self.label_list[idx])).convert("RGB")
        
        if self.transform:
            # image, label, mask = self.transform(image, label, mask)
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            #mask = F.to_tensor(mask)

        
        name = self.image_list[idx]
        # return image, label, mask, name
        return image, label, name
        
    
    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


def PGT_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        PGT_Dataset(path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class PGT_Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = os.listdir(os.path.join(image_dir)) 
        self._check_image(self.image_list) # 檢查檔案類型
        self.image_list.sort()
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        p_label = Image.open(os.path.join(self.image_dir, self.image_list[idx])).convert("RGB")
    
        if self.transform:
            p_label = self.transform(p_label)
        else:
            p_label = F.to_tensor(p_label)

        return p_label
    
    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError