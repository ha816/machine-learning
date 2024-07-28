import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from common.config import get_torch_gpu_device_if_available
from common.data.ctr_data import DisplayAdvertisingChallenge, train_test_split
from paper.ctr.deepfm.dataset import DeepFmDataset
from paper.ctr.deepfm.model import DeepFactorizationMachine
from paper.ctr.xdeepfm.model import XDeepFactorizationMachine


class XDeepFactorizationMachineController:

    def __init__(self):
        df = DisplayAdvertisingChallenge.load_df()

        sparse_feat_df = df.iloc[:, 15:]  # C1 ~ C26
        sparse_feat_grp_info = sparse_feat_df.nunique().to_dict()
        sparse_feat_df = pd.get_dummies(sparse_feat_df, dummy_na=False, dtype='int')

        dense_feat_df = df.iloc[:, 2:15]  # I1 ~ I13
        for col in dense_feat_df.columns:
            mean_val = dense_feat_df[col].mean()
            dense_feat_df[col] = dense_feat_df[col].fillna(mean_val)

        df = pd.concat([df.iloc[:, 0:2], sparse_feat_df, dense_feat_df], axis=1)
        assert df.shape[0] == df.shape[0]

        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)

        self.train_dataset = DeepFmDataset(train_df.iloc[:, 15:].values.tolist(),
                                           train_df.iloc[:, 2:15].values.tolist(),
                                           train_df.iloc[:, 1].values.tolist())

        self.test_dataset = DeepFmDataset(test_df.iloc[:, 15:].values.tolist(),
                                          test_df.iloc[:, 2:15].values.tolist(),
                                          test_df.iloc[:, 1].values.tolist())

        self.model = XDeepFactorizationMachine(list(sparse_feat_grp_info.values()),
                                               dense_feat_df.shape[1],
                                               16,
                                               [128, 64]).to(get_torch_gpu_device_if_available())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.BCELoss()

        print(f"Sparse & Dense Shape: {sparse_feat_df.shape} - {dense_feat_df.shape}")
        print(f"Sparse Feature Group Info: {sparse_feat_grp_info}")
        print(f"Preprocessed Dataframe: {df.shape}")
        print(f"Train-Test Dataframe: {train_df.shape} - {test_df.shape}")

    def train(self, epochs: int, batch_size: int = 1000):
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(1, epochs):
            loss_records = []
            for sparse_feat, dense_feat, labels in train_loader:
                predicated_ctr = self.model(sparse_feat, dense_feat)
                loss = self.criterion(predicated_ctr, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_records.append(loss.item())

            if epoch % (epochs // 20 - 1) == 0:
                print(f'Epoch [{epoch}/{epochs}], Mean Loss: {np.mean(loss_records):.4f}')

            if epoch % (epochs // 10 - 1) == 0:
                print(f'Epoch [{epoch}/{epochs}], Entering Validation Process...')
                self.test(batch_size)

        self.test(batch_size)

    def test(self, batch_size: int = 1000):
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size)

        with torch.no_grad():
            test_loss = 0
            all_labels = []
            all_preds = []

            for sparse_feat, dense_feat, labels in test_loader:
                predicated_ctr = self.model(sparse_feat, dense_feat)
                loss = self.criterion(predicated_ctr, labels)
                test_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicated_ctr.round().cpu().numpy())

            test_loss /= len(test_loader.dataset)
            accuracy = accuracy_score(all_labels, all_preds)
            print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    controller = XDeepFactorizationMachineController()
    controller.train(epochs=500, batch_size=1000)
