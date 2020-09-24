from os import path, listdir
import random

from torch.utils.data import Dataset
from PIL import Image, ImageFile


class GarbageDataset:
    def __init__(self, data_root_dir, mode, transform=None):
        # initialize
        super().__init__()

        self.data_root_dir = path.join(data_root_dir, mode)  # dataset+train, dataset+val
        self.transform = transform
        self.labels = ['Can', 'Glass', 'Plastic', 'Paper', 'Vinyl', 'PET','Paperpack']
        self.items = []  # labeled data (file+label)
        self.mode = mode

        # get labels in data directory
        # 명확하게 정해진게 아님, 출력해보니 train,val 데이터 가져오는거에 따라 라벨이 달라짐 => 결과적으로 라벨이 안맞는 현상 발생

        self.num_classes = len(self.labels)

        for label in self.labels:
            label_dir = path.join(self.data_root_dir, label)
            files = listdir(label_dir)
            for file in files:
                self.items.append([file, label])
                pass
            pass
        random.shuffle(self.items)
        print(self.items)
        pass

    def __getitem__(self, idx):
        file, label = self.items[idx]
        img = Image.open(self._get_file_path(file, label)).convert('RGB')
        label = self.labels.index(label)

        if self.transform is not None:
            img = self.transform(img)
            pass
        return img, label

    def __len__(self):
        return len(self.items)

    def _get_file_path(self, file, label):
        return path.join(self.data_root_dir, label, file)
