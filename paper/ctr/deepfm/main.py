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


class DeepFactorizationMachineController:

    def __init__(self):
        self.device = get_torch_gpu_device_if_available()

        df = DisplayAdvertisingChallenge.load_df()
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)

        sparse_feat_train_df = train_df.iloc[:, 15:]  # C1 ~ C26
        sparse_feat_grp_train_info = sparse_feat_train_df.nunique().to_dict()
        sparse_feat_train_df = pd.get_dummies(sparse_feat_train_df, dummy_na=False, dtype='int')
        dense_feat_train_df = train_df.iloc[:, 2:15]  # I1 ~ I13

        print(f"Sparse Feature Group Info: {sparse_feat_grp_train_info}")
        print(f"Train Sparse-Dense Feature Dataframe Shape: {sparse_feat_train_df.shape} - {dense_feat_train_df.shape}")

        for col in dense_feat_train_df.columns:
            mean_val = dense_feat_train_df[col].mean()
            dense_feat_train_df[col] = dense_feat_train_df[col].fillna(mean_val)

        self.model = DeepFactorizationMachine(list(sparse_feat_grp_train_info.values()),
                                              dense_feat_train_df.shape[1],
                                              16,
                                              [128, 64]).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
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

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
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

    controller = DeepFactorizationMachineController()
    controller.train(epochs=500, batch_size=1000)
