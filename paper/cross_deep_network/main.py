import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

from pandas import DataFrame
from sklearn.metrics import accuracy_score

from common.config import get_torch_gpu_device_if_available
from common.data.ctr_data import load_display_advertising_challenge_train_test_df
from paper.cross_deep_network.dataset import DcnDataset
from paper.cross_deep_network.model import CrossDeepNetwork
from torch.utils.data import DataLoader


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
            dense_feat_train_df[col].fillna(mean_value, inplace=True)

        self.model = CrossDeepNetwork(list(sparse_feat_train_cardinality_info.values()),
                                      dense_feat_train_df.shape[1],
                                      3,
                                      [64, 32]).to(self.device)

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
            dense_feat_test_df[col].fillna(mean_value, inplace=True)

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

            if epoch % (epochs // 10 - 1) == 0:
                print(f'Epoch [{epoch}/{epochs}], Mean Loss: {np.mean(loss_records):.4f}')

            if epoch % (epochs // 5 - 1) == 0:
                print(f'Epoch [{epoch}/{epochs}], Entering Validation Process...')
                self.test(batch_size)

        print(f'Entering Test Process...')
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


# def main():
#     device = get_torch_gpu_device_if_available()
#
#     df: DataFrame = load_display_advertising_challenge_df()
#
#     sparse_feat_df = df.iloc[:, 15:]  # C1 ~ C26
#     sparse_feat_cardinality_info = sparse_feat_df.nunique().to_dict()
#     sparse_feat_df = pd.get_dummies(sparse_feat_df, dummy_na=True, dtype='int')
#
#     dense_feat_df = df.iloc[:, 2:15]  # I1 ~ I13
#     for col in dense_feat_df.columns:
#         mean_value = dense_feat_df[col].mean()
#         dense_feat_df[col].fillna(mean_value, inplace=True)
#
#     dataset = DcnDataset(sparse_feat_df.values.tolist(), dense_feat_df.values.tolist(), df.iloc[:, 1].values.tolist())
#     train_loader = DataLoader(dataset, batch_size=500, shuffle=True)
#
#     model = CrossDeepNetwork(list(sparse_feat_cardinality_info.values()),
#                              dense_feat_df.shape[1],
#                              1,
#                              [64, 64])
#     model = model.to(device)
#
#     optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)
#     criterion = nn.BCELoss()
#
#     logloss = []
#     epoch_num = 10
#     iter_num = 0
#
#     model.train()
#     for epoch in range(epoch_num):
#         print('start epoch [{}/{}]'.format(epoch + 1, 5))
#
#         for sparse_feat, dense_feat, labels in train_loader:
#             iter_num += 1
#             begin_time = time.time()
#
#             predicated_ctr = model(sparse_feat, dense_feat)
#
#             loss = criterion(predicated_ctr, labels)  # Binary Cross Entropy
#             iter_loss = loss.item()
#
#             logloss.append(iter_loss)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             end_time = time.time()
#             print("epoch {}/{}, total_iter is {}, logloss is {:.2f}, cost time is {:.2f}s".format(epoch + 1, epoch_num,
#                                                                                                   iter_num, iter_loss,
#                                                                                                   end_time - begin_time))


if __name__ == '__main__':
    controller = CrossDeepNetworkController()
    controller.train(epochs=100, batch_size=500)
