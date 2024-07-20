import torch
from torch.utils.data import Dataset


class DcnDataset(Dataset):

    def __init__(self, sparse_feat, dense_feat, labels):
        super(DcnDataset, self).__init__()

        assert len(sparse_feat) == len(dense_feat) == len(labels)
        self.sparse_feat = sparse_feat
        self.dense_feat = dense_feat
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        print(index)
        return torch.tensor(self.sparse_feat[index], dtype=torch.float), \
            torch.tensor(self.dense_feat[index], dtype=torch.float), \
            torch.tensor(self.labels[index], dtype=torch.float)
