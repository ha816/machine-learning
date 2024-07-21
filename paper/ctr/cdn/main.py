import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from common.config import get_torch_gpu_device_if_available
from common.data.ctr_data import load_display_advertising_challenge_train_test_df
from paper.cross_deep_network.dataset import DcnDataset
from paper.cross_deep_network.model import CrossDeepNetwork


class CrossDeepNetworkController:

    def __init__(self):
        self.device = get_torch_gpu_device_if_available()

        train_df, test_df = load_display_advertising_challenge_train_test_df()

        sparse_feat_train_df = train_df.iloc[:, 15:]  # C1 ~ C26
        sparse_feat_train_cardinality_info = sparse_feat_train_df.nunique().to_dict()
        sparse_feat_train_df = pd.get_dummies(sparse_feat_train_df, dummy_na=True, dtype='int')

        dense_feat_train_df = train_df.iloc[:, 2:15]  # I1 ~ I13
        for col in dense_feat_train_df.columns:
            mean_value = dense_feat_train_df[col].mean()
            dense_feat_train_df[col] = dense_feat_train_df[col].fillna(mean_value)

        self.model = CrossDeepNetwork(list(sparse_feat_train_cardinality_info.values()),
                                      dense_feat_train_df.shape[1],
                                      4,
                                      [128, 64]).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.criterion = nn.BCELoss()

        self.train_dataset = DcnDataset(sparse_feat_train_df.values.tolist(),
                                        dense_feat_train_df.values.tolist(),
                                        train_df.iloc[:, 1].values.tolist())

        sparse_feat_test_df = test_df.iloc[:, 15:]  # C1 ~ C26
        sparse_feat_test_df = pd.get_dummies(sparse_feat_test_df, dummy_na=True, dtype='int')

        dense_feat_test_df = test_df.iloc[:, 2:15]  # I1 ~ I13
        for col in dense_feat_test_df.columns:
            mean_value = dense_feat_test_df[col].mean()
            dense_feat_test_df[col] = dense_feat_test_df[col].fillna(mean_value)

        self.test_dataset = DcnDataset(sparse_feat_test_df.values.tolist(),
                                       dense_feat_test_df.values.tolist(),
                                       test_df.iloc[:, 1].values.tolist())

    def train(self, epochs: int, batch_size: int = 500):

        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs):

            loss_records = []
            for sparse_feat, dense_feat, labels in train_loader:
                self.optimizer.zero_grad()

                predicated_ctr = self.model(sparse_feat, dense_feat)
                loss = self.criterion(predicated_ctr, labels)
                loss.backward()
                self.optimizer.step()

                iter_loss = loss.item()
                loss_records.append(iter_loss)

            if epoch % (epochs // 20 - 1) == 0:
                print(f'Epoch [{epoch}/{epochs}], Mean Loss: {np.mean(loss_records):.4f}')

            if epoch % (epochs // 10 - 1) == 0:
                print(f'Epoch [{epoch}/{epochs}], Entering Validation Process...')
                self.test(batch_size)

        self.test(batch_size)

    def test(self, batch_size: int = 500):
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
    controller = CrossDeepNetworkController()
    controller.train(epochs=100, batch_size=500)
