from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import os
import fnmatch
import librosa
import librosa.display
import numpy as np
import scipy.io as scio


def load_data(data_folder, batch_size, sr=12*1024, train=True, graph=False, num_workers=4, ratio=0.7, **kwargs):
    data_file_type = os.path.splitext(os.listdir(os.path.join(data_folder, os.listdir(data_folder)[0]))[0])[1]
    if data_file_type == '.jpg':
        transform = {
            'train': transforms.Compose(
                [transforms.Resize([256, 256]),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]),
            'test': transforms.Compose(
                [transforms.Resize([224, 224]),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
        }
        data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
        # data = random_split(data_all, [int(len(data_all)*ratio), int(len(data_all))-int(len(data_all)*ratio)])
        data_loader = get_data_loader(data, batch_size=batch_size,
                                      shuffle=True if train else False,
                                      num_workers=num_workers, **kwargs,
                                      drop_last=True if train else False,
                                      pin_memory=True)
        # n_class = len(data.classes)
    if data_file_type == '.wav':
        data = VibDataset(data_folder, sr=sr)
        train_data, test_data = random_split(data, [int(len(data)*ratio), int(len(data))-int(len(data)*ratio)])
        data_loader = get_data_loader(train_data if train else test_data,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      shuffle=True if train else False,  **kwargs,
                                      drop_last=True)

    if data_file_type == '.mat':
        data = MatDataset(data_folder)
        train_data, test_data = random_split(data,
                                             [int(len(data) * ratio), int(len(data)) - int(len(data) * ratio)])
        data_loader = get_data_loader(train_data if train else test_data,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      shuffle=True if train else False, **kwargs,
                                      drop_last=True)
        # data = VibDataset(data_folder, sr=sr)
        # data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, sampler=None)
    return data_loader

def get_data_loader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=0, infinite_data_loader=False, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           drop_last=drop_last,
                                           num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, **kwargs)


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0


class VibDataset(Dataset):
    def __init__(self, data_dir, sr, dimension=4096):
        self.data_dir = data_dir
        self.sr = sr
        self.dim = dimension

        # 获取wav文件列表
        self.file_list = []
        for root, dirname, filenames in os.walk(data_dir):
            for filename in fnmatch.filter(filenames, "*.wav"):
                self.file_list.append(os.path.join(root, filename))

    def __getitem__(self, item):
        filename = self.file_list[item]
        wb_wav, _ = librosa.load(filename, sr=self.sr, mono=False)
        wb_wav = np.expand_dims(wb_wav, axis=0)
        path, file_name = os.path.split(filename)
        label = int(path[-1])

        # librosa.display.waveshow(wb_wav[0])
        # plt.show()
        # librosa.display.waveshow(wb_wav[1])
        # plt.show()

        return wb_wav, label, filename

    def __len__(self):

        return len(self.file_list)


class MatDataset(Dataset):
    def __init__(self, data_dir, dimension=4096):
        self.data_dir = data_dir
        self.dim = dimension
        # 获取mat文件列表
        self.file_list = []
        for root, dirname, filenames in os.walk(data_dir):
            for filename in fnmatch.filter(filenames, "*.mat"):
                self.file_list.append(os.path.join(root, filename))

    def __getitem__(self, item):
        filename = self.file_list[item]
        wb_mat = scio.loadmat(filename)['signal_sample']
        wb_mat = wb_mat.astype(np.float32)
        wb_mat = np.squeeze(wb_mat)
        wb_mat = np.expand_dims(wb_mat, axis=0)
        wb_mat = np.expand_dims(wb_mat, axis=0)
        path, file_name = os.path.split(filename)
        label = int(path[-1])

        return wb_mat, label, filename

    def __len__(self):

        return len(self.file_list)

# def main():
#     Dataset = VibDataset(r"D:\DATABASE\ZXJ_GD\sample\10-B1-4_CH1-8", sr=int(25e3))
#     for i, data in enumerate(Dataset):
#         vib_wav, label, filename = data
#         print(i, vib_wav.shape, filename)
#         if i == 3:
#             break
#     batch = DataLoader(Dataset, batch_size=64, shuffle=False, sampler=None)
#     DataLoader(Dataset, batch_size=32, shuffle=False, sampler=None)
#
# if __name__ == '__main__':
#     main()

