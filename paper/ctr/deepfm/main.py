import pdb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from common.config import get_torch_gpu_device_if_available
from common.data.ctr_data import load_display_advertising_challenge_train_test_df
from paper.ctr.deepfm.dataset import DeepFmDataset
from paper.ctr.deepfm.model import DeepFactorizationMachine


class DeepFactorizationMachineController:

    def __init__(self):
        self.device = get_torch_gpu_device_if_available()

        train_df, test_df = load_display_advertising_challenge_train_test_df()

        sparse_feat_train_df = train_df.iloc[:, 15:]  # C1 ~ C26
        sparse_feat_grp_train_info = sparse_feat_train_df.nunique().to_dict()
        sparse_feat_train_df = pd.get_dummies(sparse_feat_train_df, dummy_na=False, dtype='int')

        dense_feat_train_df = train_df.iloc[:, 2:15]  # I1 ~ I13

        print(f"Sparse Feature Group Info: {sparse_feat_grp_train_info}")
        print(f"Train Sparse-Dense Feature Dataframe Shape: {sparse_feat_train_df.shape} - {dense_feat_train_df.shape}")

        # TODO MinMax scale
        # TODO Z-Norm 공통 함수로 뽑아내기
        # dense_feat_train_df = (dense_feat_train_df - dense_feat_train_df.min()) / (
        #             dense_feat_train_df.max() - dense_feat_train_df.min())

        # dense_feat_train_df = (dense_feat_train_df - dense_feat_train_df.mean()) / dense_feat_train_df.std()
        # print(dense_feat_train_df.describe())

        for col in dense_feat_train_df.columns:
            mean_val = dense_feat_train_df[col].mean()
            # std_val = dense_feat_train_df[col].std()
            # dense_feat_train_df[col] = (dense_feat_train_df[col] - mean_val) / std_val
            dense_feat_train_df[col] = dense_feat_train_df[col].fillna(mean_val)

        self.model = DeepFactorizationMachine(list(sparse_feat_grp_train_info.values()),
                                              dense_feat_train_df.shape[1],
                                              16,
                                              [128, 64]).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()

        self.train_dataset = DeepFmDataset(sparse_feat_train_df.values.tolist(),
                                           dense_feat_train_df.values.tolist(),
                                           train_df.iloc[:, 1].values.tolist())

        sparse_feat_test_df = test_df.iloc[:, 15:]  # C1 ~ C26
        sparse_feat_test_df = pd.get_dummies(sparse_feat_test_df, dummy_na=True, dtype='int')

        dense_feat_test_df = test_df.iloc[:, 2:15]  # I1 ~ I13
        for col in dense_feat_test_df.columns:
            mean_val = dense_feat_test_df[col].mean()
            dense_feat_test_df[col] = dense_feat_test_df[col].fillna(mean_val)

        self.test_dataset = DeepFmDataset(sparse_feat_test_df.values.tolist(),
                                          dense_feat_test_df.values.tolist(),
                                          test_df.iloc[:, 1].values.tolist())
        self.model.to(self.device)

    def train(self, epochs: int, batch_size: int = 1000):
        self.model.train()

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)  #
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(1, epochs):
            loss_records = []
            for sparse_feat, dense_feat, labels in train_loader:

                predicated_ctr = self.model(sparse_feat, dense_feat)
                loss = self.criterion(predicated_ctr, labels)

                self.optimizer.zero_grad()
                try:
                    loss.backward()
                    iter_loss = loss.item()
                except Exception as err:
                    print(f"ERROR: {err}, LOSS: {loss.item()}")
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            print(
                                f"{name}: {param.grad.norm()}, {param.shape}, {param.requires_grad}, {param.isnan().any()}, {param.isinf().any()}")
                        else:
                            print(
                                f"{name}: GRAD NONE, {param.shape}, {param.requires_grad}, {param.isnan().any()}, {param.isinf().any()}")
                    continue

                self.optimizer.step()
                loss_records.append(iter_loss)

            if epoch % (epochs // 20 - 1) == 0:
                print(f'Epoch [{epoch}/{epochs}], Mean Loss: {np.mean(loss_records):.4f}')

            # if epoch % (epochs // 10 - 1) == 0:
            #     print(f'Epoch [{epoch}/{epochs}], Entering Validation Process...')
            #     self.test(batch_size)

        # self.test(batch_size)

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
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    controller = DeepFactorizationMachineController()
    controller.train(epochs=1000, batch_size=1000)  # batch_size를 2000으로 하면 문제가 없는데...?
    # 500으로 suffle true시 문제가 있네?
    # 300으로 하면 17먼째에서 이슈?
    # 500으로 하면 25번째에서 이슈?
    # learning rate를 약간 높임

    # if torch.isnan(sparse_feat).any() or torch.isnan(dense_feat).any(): # 해당 없음
    #     continue
    # if torch.isinf(sparse_feat).any() or torch.isinf(dense_feat).any(): # 해당 없음
    #     continue
