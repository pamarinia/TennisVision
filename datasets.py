from torch.utils.data import Dataset
import pandas as pd
import os


class TrackNetDataset(Dataset):
    def __init__(self, mode):
        self.path_dataset = 'datasets/TrackNet'
        assert mode in ['train', 'val'], 'incorrect mode'
        self.data = pd.read_csv(os.path.join(self.path_dataset, 'labels_{}.csv'.format(mode)))
        print('mode = {}, samples = {}'.format(mode, self.data.shape[0]))


if __name__ == '__main__':
    dataset = TrackNetDataset('train')