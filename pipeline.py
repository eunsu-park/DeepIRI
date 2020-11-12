import torch
from torch.utils import data
import numpy as np
import torchvision.transforms as T
from glob import glob
import os

class Normalizer:
    def __call__(self, data):
        return (data/50.)-1.

class DeNormalizer:
    def __call__(self, data):
        return (data+1.)*50.

class SnapMaker:
    def __call__(self, data):
        tmp = (data+1.)*(255/2.)
        snap = np.clip(tmp, 0, 255).astype(np.uint8)
        return snap

def get_transforms():
    transforms = []
    transforms.append(Normalizer())    
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

class CustomDataset:
    def __init__(self, opt):
        self.is_train = opt.is_train
        if self.is_train == True :
            pattern_inp = os.path.join(opt.root_data, 'train', opt.name_inp, '*.npy')
            self.list_inp = sorted(glob(pattern_inp))
            pattern_tar = os.path.join(opt.root_data, 'train', opt.name_tar, '*.npy')
            self.list_tar = sorted(glob(pattern_tar))
            assert len(self.list_inp) == len(self.list_tar)
            self.nb_data = len(self.list_inp)
        else :
            pattern_inp = os.path.join(opt.root_data, 'test', opt.name_inp, '*.npy')
            self.list_inp = sorted(glob(pattern_inp))
            self.nb_data = len(self.list_inp)
        self.norm = Normalizer()

    def __len__(self):
        return self.nb_data

    def __getitem__(self, idx):
        inp = np.load(self.list_inp[idx])[None,:,:]
        inp = self.norm(inp)
        inp = torch.from_numpy(inp)
        if self.is_train == True :
            tar = np.load(self.list_tar[idx])[None,:,:]
            tar = self.norm(tar)
            tar = torch.from_numpy(tar)
        else :
            tar = torch.zeros_like(inp, dtype=torch.float)
        return inp, tar

def get_data_loader(opt):
    dataset = CustomDataset(opt)
    dataloader = data.DataLoader(dataset=dataset, batch_size=opt.batch_size,
                                 shuffle=opt.is_train, num_workers=opt.num_workers)
    return dataloader