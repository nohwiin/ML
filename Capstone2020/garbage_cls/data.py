from torchvision import transforms
from torch.utils.data import DataLoader

from garbage_cls.dataset import GarbageDataset

_dataloader_params_dict = {
    'train': {
        'batch_size':128,
        'num_workers': 4,
        'shuffle': True # dataset에서 랜덤으로 index 뽑아서 batch 만듬
    },
    'val': {
        'batch_size': 128,
        'num_workers': 4,
        'shuffle': False # dataset에서 0부터 index 뽑아서 batch 만듬
    },
    'test': {
        'batch_size': 128,
        'num_workers': 4,
        'shuffle': False
    }
}

# dataset을 batch size에 맞춰서 sampling 하는것
def get_dataset(data_root_dir, mode, transform=None):
    return GarbageDataset(data_root_dir=data_root_dir,
                          mode=mode,
                          transform=transform)


def get_dataloader(dataset, mode):
    params = _dataloader_params_dict[mode]
    return DataLoader(dataset, **params)
