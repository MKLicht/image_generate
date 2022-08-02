import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CatData(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.img_list = []
        self.img_list0 = glob.glob(root + '/dataset-part1/*.png')
        self.img_list1 = glob.glob(root + '/dataset-part2/*.png')
        self.img_list2 = glob.glob(root + '/dataset-part3/*.png')
        self.img_list = self.img_list0 + self.img_list1 + self.img_list2

    def __getitem__(self, index):
        img = Image.open(self.img_list[index % len(self.img_list)])
        item = self.transform(img)

        return item

    def __len__(self):     
        return len(self.img_list)
