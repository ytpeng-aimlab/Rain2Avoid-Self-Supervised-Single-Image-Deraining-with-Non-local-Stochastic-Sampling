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
    



class FRandomCrop(transforms.RandomCrop):
    def __call__(self, image, label, mask, p_label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
            mask = F.pad(mask, self.padding, self.fill, self.padding_mode)
            p_label = F.pad(p_label, self.padding, self.fill, self.padding_mode)
        
        # pad the width if needed
        if image.size[0] < self.size[1]:
            p = self.size[1] - image.size[0]
            image = F.pad(image, (p, 0), self.fill, self.padding_mode)
            label = F.pad(label, (p, 0), self.fill, self.padding_mode)
            mask = F.pad(mask, (p, 0), self.fill, self.padding_mode)
            p_label = F.pad(p_label, (p, 0), self.fill, self.padding_mode)
        
        # pad the height if needed
        if image.size[1] < self.size[0]:
            p = self.size[0] - image.size[1]
            image = F.pad(image, (0, p), self.fill, self.padding_mode)
            label = F.pad(label, (0, p), self.fill, self.padding_mode)
            mask = F.pad(mask, (0, p), self.fill, self.padding_mode)
            p_label = F.pad(p_label, (0, p), self.fill, self.padding_mode)
        
        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w), F.crop(mask, i, j, h, w), F.crop(p_label, i, j, h, w)


class FCompose(transforms.Compose):
    def __call__(self, image, label, mask, p_label):
        for t in self.transforms:
            image, label, mask, p_label = t(image, label, mask, p_label)
        return image, label, mask, p_label

class FRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label, mask, p_label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label), F.hflip(mask), F.hflip(p_label)
        return img, label, mask, p_label

class FToTensor(transforms.ToTensor):
    def __call__(self, pic, label, mask, p_label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label), F.to_tensor(mask),F.to_tensor(p_label)

        