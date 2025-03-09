import os

from torch.utils.data import Dataset

class MaestroDataset(Dataset):
    def __init__(self, data_dir, split):
        data_path = os.path.join(data_dir, f'{split}.txt')
        self.data = []
        print("path", data_path)
        with open(data_path, 'r') as f:
            for line in f:
                tokens = [int(x) for x in line.strip().split()]
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': self.data[idx], 'labels': self.data[idx]}