import json
from torch.utils.data import Dataset


class DatasetEnzyme(Dataset):
    def __init__(self, data_file, label2id_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        with open(label2id_file, 'r') as f:
            self.label2id = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item['label'] = self.label2id[item['label']]
        return item.values()