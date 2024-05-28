from torch.utils.data import Dataset
import json
import torch

class EnzymeDataset(Dataset):
    def __init__(self, data_file, label2id_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        with open(label2id_file, 'r') as f:
            self.label2id = json.load(f)

    def remove_non_enzymes(self):
        self.data = [item for item in self.data if item['label'] != '0']
        self.label2id.pop('0')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item['label'] = self.label2id[item['label']]
        return item.values()


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        if len(batch[0]) == 3:
            ids, labels, seqs = zip(*batch)
        else:
            ids, labels, seqs, structure_features = zip(*batch)
        inputs = self.tokenizer(list(seqs), return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        labels = torch.tensor(list(labels), dtype=torch.long)
        inputs['labels'] = labels
        # for esm model
        # return inputs

        # for point net model
        # return {'structures': torch.tensor(structure_features).permute(0, 2, 1), 'labels': labels}

        # for esm_bbc model with structure features
        structure_features = list(structure_features)
        for idx, item in enumerate(structure_features):
            if item is None:
                structure_features[idx] = [[float(0)]*4]*1000

        inputs['structures'] = torch.tensor(structure_features).permute(0, 2, 1)

        return inputs


if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    # from transformers import AutoTokenizer, EsmForSequenceClassification
    # tokenizer = AutoTokenizer.from_pretrained('tokenizer-esm2_t30_150M_UR50D')
    # collate_fn = Collate(tokenizer)

    dataset_train = EnzymeDataset('dataset_swiss_prot_folds/dataset_swiss_prot_fold0/train.json', 'dataset_swiss_prot_folds/dataset_swiss_prot_fold0/label2id.json')
    print(len(dataset_train))
    dataset_train.remove_non_enzymes()
    print(len(dataset_train))
    # dataloader_train = DataLoader(dataset_train, batch_size=4, collate_fn=collate_fn)
    # for batch in dataloader_train:
    #     print(batch['structures'].shape)
    #     print(batch['labels'].shape)
    #     break

