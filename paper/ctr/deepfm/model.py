# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
import math
import pdb

import numpy as np
import torch
import torch.nn as nn


class DeepFactorizationMachine(nn.Module):
    """Instantiates the DeepFM Network architecture.
    """

    def __init__(self,
                 sparse_feat_cardinality_list: list[int],
                 n_dense_feat: int,
                 emb_size: int,
                 deep_layers: list[int]):
        super(DeepFactorizationMachine, self).__init__()

        self.sparse_feat_cardinality_list = sparse_feat_cardinality_list

        self.fm = FactorizationMachine(sparse_feat_cardinality_list, n_dense_feat)

        self.sparse_emb_list = nn.ModuleList([nn.Embedding(field_size, emb_size, scale_grad_by_freq=True)
                                              for field_size in sparse_feat_cardinality_list])

        for sparse_emb in self.sparse_emb_list:
            torch.nn.init.xavier_uniform_(sparse_emb.weight)
            # torch.nn.init.normal_(sparse_emb.weight, mean=0.0, std=0.0001)

        n_spare_feat_grp = len(sparse_feat_cardinality_list)
        self.dnn = DeepNeuralNetwork(n_spare_feat_grp * emb_size + n_dense_feat, deep_layers)

    def forward(self, sparse_feat, dense_feat):
        """
        :param sparse_feat: [B, S.F]
        :param dense_feat: [B, D.F]
        :return:
        """
        spare_feat_grp_list = self.get_spare_feat_grp_list(sparse_feat)

        sparse_feat_grp_emb_list = []
        for x, emb in zip(spare_feat_grp_list, self.sparse_emb_list):
            w_x = torch.mm(x, emb.weight)
            sparse_feat_grp_emb_list.append(w_x)

        sparse_feat_emb = torch.stack(sparse_feat_grp_emb_list, dim=1)  # [B, S.G.F, E]
        fm_term = self.fm(spare_feat_grp_list, sparse_feat_emb, dense_feat)
        # 이유는 모르겠으나 fm_term nan이 들어감 ㅋㅋ....

        dnn_feat_emb = torch.concat((torch.flatten(sparse_feat_emb, start_dim=1), dense_feat), dim=-1)
        dnn_term = self.dnn(dnn_feat_emb)

        y_pred = torch.sigmoid(fm_term + dnn_term)

        print(f"FM + DNN : {fm_term.mean()} / {dnn_term.mean()}, Y : {y_pred.sum()}")

        if torch.isnan(y_pred).any():
            print("DESTROY")

        return y_pred.squeeze(-1)

    def get_spare_feat_grp_list(self, sparse_feat):
        spare_feat_grp_list = []
        begin = 0
        for field_size in self.sparse_feat_cardinality_list:
            end = begin + field_size
            x_i = sparse_feat[:, begin: end]
            assert x_i.shape[1] == field_size

            spare_feat_grp_list.append(x_i)
            begin = end
        return spare_feat_grp_list


class FactorizationMachine(nn.Module):

    def __init__(self,
                 sparse_feat_cardinality_list: list[int],
                 n_dense_feat: int):
        super(FactorizationMachine, self).__init__()

        n_sparse_grp_feat = len(sparse_feat_cardinality_list)

        emb_list = []
        for field_size in sparse_feat_cardinality_list:
            emb = nn.Embedding(field_size, 1, scale_grad_by_freq=True) # scale 하면 괜찮나??
            torch.nn.init.xavier_uniform_(emb.weight)
            # torch.nn.init.normal_(emb.weight, mean=0.0, std=0.0001) # 왜
            emb_list.append(emb)

        self.sparse_emb_list_for_linear = nn.ModuleList(emb_list) # 왜 여기서도 null?
        # self.sparse_emb_list_for_linear = nn.ModuleList([nn.Embedding(field_size, 1, max_norm=1.0)
        #                                                  for field_size in sparse_feat_cardinality_list])

        # self.sparse_linear_scale_factor = math.sqrt(np.sum(sparse_feat_cardinality_list))

        self.linear = nn.Linear(n_sparse_grp_feat + n_dense_feat, 1)
        for name, param in self.linear.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            # param.bias.data.fill_(0.01)
            # torch.nn.init.normal_(param, mean=0.0, std=0.0001)
        # torch.nn.init.normal_(self.linear.bias, mean=0.0, std=0.00001)
        # nn.Linear의 파라미터 초기화 self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # torch.empty가 상황에 따라 nan weight를 가지는 경우가 생김

        self.pfm = PairwiseFactorizationMachine()

    def forward(self, sparse_feat_grp_list, sparse_feat_emb, dense_feat):
        """
        sparse_feat_grp_list: input feat 값을 그대로 보존(Not Embeded)
        sparse_feat_emb: [batch_size, sparse_group_size, embedding_size]
        """
        sparse_linear_emb_list = []
        for idx, (sparse_feat_grp, linear_emb) in enumerate(zip(sparse_feat_grp_list, self.sparse_emb_list_for_linear)):
            if torch.isnan(linear_emb.weight).any():
                print(f"{idx} : {sparse_feat_grp.shape}: {torch.isnan(linear_emb.weight).sum()}")
                # 25? 왜 유독 sparse 25에서만 그러지?

            sparse_linear_emb = torch.mm(sparse_feat_grp, linear_emb.weight)  # [1]
            # sparse_linear_emb = torch.mm(sparse_feat_grp, linear_emb.weight)  # [1]
            sparse_linear_emb_list.append(sparse_linear_emb)

        sparse_linear_emb = torch.concat(sparse_linear_emb_list, dim=-1)
        fm_linear_input = torch.concat((sparse_linear_emb, dense_feat), dim=-1)  # [B, S.G.F + D.F]

        # print(f"MAX:{fm_linear_input.max()}")  # min-max scaling으로 1로 만듬
        fm_linear_and_bias = self.linear(fm_linear_input)  #

        # fm_linear_and_bias = self.linear(fm_linear_input) / self.sparse_linear_scale_factor

        fm_pairwise = self.pfm(sparse_feat_emb)
        fm_output = fm_linear_and_bias + fm_pairwise

        # print(f"FM {fm_linear_and_bias.mean()} / {fm_pairwise.mean()} / {fm_output.mean()}")
        return fm_output


class DeepNeuralNetwork(nn.Module):
    """
    All of the layer in this module are full-connection layers
    """

    def __init__(self, n_input_feat, layers: list):
        """
        :param n_input_feat: total num of input_feature, including of the embedding feature and dense feature
        :param layers: a list contains the num of each hidden layer's units
        """
        super(DeepNeuralNetwork, self).__init__()
        fc_layers = [nn.Linear(n_input_feat, layers[0]),
                     nn.BatchNorm1d(layers[0], affine=False),
                     nn.ReLU(inplace=True)]

        for i in range(1, len(layers)):
            fc_layers.append(nn.Linear(layers[i - 1], layers[i]))
            fc_layers.append(nn.BatchNorm1d(layers[i], affine=False))
            fc_layers.append(nn.Sigmoid())

        fc_layers.append(nn.Linear(layers[-1], 1, bias=False))
        self.deep = nn.Sequential(*fc_layers)

    def forward(self, x):
        dense_output = self.deep(x)
        return dense_output


class PairwiseFactorizationMachine(nn.Module):
    """
    Factorization Machine models pairwise (order-2) feature interactions
    *Without linear term and bias*
    pairwise (order-2) feature interactions refer to the interactions  between every possible pair of features in the dataset.
    """

    def __init__(self):
        super(PairwiseFactorizationMachine, self).__init__()

    def forward(self, x):
        # sparse feat 정보도 활용
        """
        Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
        Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
        """
        square_of_sum = torch.pow(torch.sum(x, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(x * x, dim=1, keepdim=True)

        cross_term = square_of_sum - sum_of_square  # (batch_size,1,embedding_size)
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)  # (batch_size,1)
        return cross_term
