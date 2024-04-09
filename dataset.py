from torchvision.datasets import ImageFolder, CIFAR100, CIFAR10
import torchvision.transforms as T
from PIL import Image

class UnlabelSTL10(ImageFolder):
    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class UnlabelCIFAR100(CIFAR100):
    def __getitem__(self, index: int):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

class UnlabelCIFAR10(CIFAR10):
    def __getitem__(self, index: int):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img


class StlPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = T.Compose([
                    T.RandomApply(
                        [T.ColorJitter(brightness=0.4, contrast=0.4, 
                                                saturation=0.4, hue=0.1)], 
                        p=0.8
                    ),
                    T.RandomGrayscale(p=0.1),
                    T.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                ])
        else:
            self.transform = T.Compose([
                    T.Resize(70, interpolation=Image.BICUBIC),
                    T.CenterCrop(64),
                    T.ToTensor(),
                    T.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                ])
        self.pair_transform = pair_transform
        
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

class CIFARPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = T.Compose([
                    T.RandomApply(
                        [T.ColorJitter(brightness=0.4, contrast=0.4, 
                                                saturation=0.4, hue=0.1)], 
                        p=0.8
                    ),
                    T.RandomGrayscale(p=0.1),
                    T.RandomResizedCrop(
                        32,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
        else:
            self.transform = T.Compose([
                    T.Resize(40, interpolation=Image.BICUBIC),
                    T.CenterCrop(32),
                    T.ToTensor(),
                    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
        self.pair_transform = pair_transform
        
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

class ImageNetPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
            if train_transform is True:
                self.transform = T.Compose([
                        T.RandomApply(
                            [T.ColorJitter(brightness=0.4, contrast=0.4, 
                                                    saturation=0.4, hue=0.1)], 
                            p=0.8
                        ),
                        T.RandomGrayscale(p=0.1),
                        T.RandomResizedCrop(
                            224,
                            scale=(0.08, 1.0),
                            ratio=(0.75, (4 / 3)),
                            interpolation=Image.BICUBIC,
                        ),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
            else:
                self.transform = T.Compose([
                        T.Resize(256, interpolation=Image.BICUBIC),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
            self.pair_transform = pair_transform

    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

class TinyImPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
            if train_transform is True:
                self.transform = T.Compose([
                        T.RandomApply(
                            [T.ColorJitter(brightness=0.4, contrast=0.4, 
                                                    saturation=0.4, hue=0.1)], 
                            p=0.8
                        ),
                        T.RandomGrayscale(p=0.1),
                        T.RandomResizedCrop(
                            64,
                            scale=(0.08, 1.0),
                            ratio=(0.75, (4 / 3)),
                            interpolation=Image.BICUBIC,
                        ),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ToTensor(),
                        T.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                    ])
            else:
                self.transform = T.Compose([
                        T.Resize(70, interpolation=Image.BICUBIC),
                        T.CenterCrop(64),
                        T.ToTensor(),
                        T.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                    ])
            self.pair_transform = pair_transform

    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)