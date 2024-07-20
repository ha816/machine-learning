import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

from pandas import DataFrame

from common.data.ctr_data import load_display_advertising_challenge_df
from paper.cross_deep_network.dataset import DcnDataset
from paper.cross_deep_network.model import CrossDeepNetwork
from torch.utils.data import DataLoader

# gpu_device = torch.device("cuda")
# cpu_device = torch.device("cpu")


def main():
    # 환경, 설정 파라미터 정리

    df: DataFrame = load_display_advertising_challenge_df()

    # df.columns[15:] C1 ~ C26
    # df.columns[2:14] I1 ~ I13

    sparse_feat_df: DataFrame = df.iloc[:, 15:]

    sparse_feat_df_encoded_with_nan = pd.get_dummies(sparse_feat_df, dummy_na=False, dtype='int')
    # nan은 없는 필드로 취급
    # print(sparse_feat_df_encoded_with_nan)  # [1999][13105]
    dense_feat_df = df.iloc[:, 2:15]

    dataset = DcnDataset(sparse_feat_df_encoded_with_nan.values.tolist(), dense_feat_df.values.tolist(), df.iloc[:, 1].values.tolist())
    train_loader = DataLoader(dataset, batch_size=500, shuffle=True)

    sparse_feat_cardinality_info = sparse_feat_df.nunique().to_dict()
    model = CrossDeepNetwork(list(sparse_feat_cardinality_info.values()),
                             dense_feat_df.shape[1],
                             2,
                             [128, 64]
                             )

    # print(df_encoded_with_nan.head()) # 2개만의 필드만 해도 200개가 늘어나는데 ㅋㅋㅋ.....우짜

    # model = model.to(gpu_device)
    optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)
    criterion = nn.BCELoss()

    logloss = []
    epoch_num = 10
    iter_num = 0

    for epoch in range(epoch_num):
        print('starit epoch [{}/{}]'.format(epoch + 1, 5))
        model.train()
        for sparse_feat, dense_feat, labels in train_loader:

            iter_num += 1
            begin_time = time.time()

            predicated_ctr = model(sparse_feat, dense_feat)

            loss = criterion(predicated_ctr, labels)  # Binary Cross Entropy
            iter_loss = loss.item()

            logloss.append(iter_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            print("epoch {}/{}, total_iter is {}, logloss is {:.2f}, cost time is {:.2f}s".format(epoch + 1, epoch_num,
                                                                                                  iter_num, iter_loss,
                                                                                                  end_time - begin_time))
            if iter_num % 20 == 0:
                total_loss = np.mean(logloss)
                logloss = []
                summary.add_scalar('logloss', total_loss, iter_num)

            if iter_num % 2000 == 0:
                save_dir = '../../model/cdnet' + str(iter_num) + '.pkl'
                torch.save(model.state_dict(), save_dir)
                auc_score, bias_score = test()
                summary.add_scalar('auc', auc_score, iter_num)
                summary.add_scalar('bias', bias_score, iter_num)
                model.train()


if __name__ == '__main__':
    main()
