import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class RandomCrop(transforms.RandomCrop):
    def __call__(self, image, label, mask, sdr):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
            mask = F.pad(mask, self.padding, self.fill, self.padding_mode)
            sdr = F.pad(sdr, self.padding, self.fill, self.padding_mode)
        
        if image.size[0] < self.size[1]:
            p = self.size[1] - image.size[0]
            image = F.pad(image, (p, 0), self.fill, self.padding_mode)
            label = F.pad(label, (p, 0), self.fill, self.padding_mode)
            mask = F.pad(mask, (p, 0), self.fill, self.padding_mode)
            sdr = F.pad(sdr, (p, 0), self.fill, self.padding_mode)
        
        if image.size[1] < self.size[0]:
            p = self.size[0] - image.size[1]
            image = F.pad(image, (0, p), self.fill, self.padding_mode)
            label = F.pad(label, (0, p), self.fill, self.padding_mode)
            mask = F.pad(mask, (0, p), self.fill, self.padding_mode)
            sdr = F.pad(sdr, (0, p), self.fill, self.padding_mode)
        
        i, j, h, w = self.get_params(image, self.size)
        
        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w), F.crop(mask, i, j, h, w), F.crop(sdr, i, j, h, w)

class Compose(transforms.Compose):
    def __call__(self, image, label, mask, sdr):
        for t in self.transforms:
            image, label, mask, sdr = t(image, label, mask, sdr)
        return image, label, mask, sdr

class RandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label, mask, sdr):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label), F.hflip(mask), F.hflip(sdr)
        return img, label, mask, sdr

class ToTensor(transforms.ToTensor):
    def __call__(self, pic, label, mask, sdr):
        return F.to_tensor(pic), F.to_tensor(label), F.to_tensor(mask), F.to_tensor(sdr)