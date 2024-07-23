# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
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
                 deep_layers: list[int]
                 ):
        super(DeepFactorizationMachine, self).__init__()

        # self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0

        self.sparse_feat_cardinality_list = sparse_feat_cardinality_list

        n_spare_group_feat = len(sparse_feat_cardinality_list)
        n_input_feat = np.sum(sparse_feat_cardinality_list) + n_dense_feat

        self.fm = FactorizationMachine(sparse_feat_cardinality_list, n_dense_feat, emb_size)

        self.sparse_emb_list = nn.ModuleList([nn.Embedding(field_size, emb_size)
                                              for field_size in sparse_feat_cardinality_list])
        self.dense_weight = nn.Parameter(torch.Tensor(n_spare_group_feat + n_dense_feat, 1))

        # self.fm_linear = nn.Linear(n_input_feat, 1)
        self.pfm = PairwiseFactorizationMachine()

        self.dnn = DeepNeuralNetwork(n_input_feat, deep_layers)

        # self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
        #                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
        #                init_std=init_std, device=device)

        # self.dnn_linear = nn.Linear(
        #     dnn_hidden_units[-1], 1, bias=False).to(device)  # 마지막 결과인 실수를 뽑기 위해서 한 단계를 더 둠

        # self.add_regularization_weight(
        #     filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        # self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

    def forward(self, sparse_feat, dense_feat):
        """
        :param sparse_feat: [B, S.F]
        :param dense_feat: [B, D.F]
        :return:
        """
        # sparse_emb_list을 이용해서 sparse_input_feat과 곱해서 나온 갚을 임베딩으로 활용

        fm_term = self.fm(sparse_feat, dense_feat)

        # dense 피처 에 하나씩 weight
        # nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1)
        # sparse 피처에 embedding
        # embedding_dict = nn.ModuleDict(
        #     {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1,
        #                                        sparse=sparse)
        #      for feat in sparse_feature_columns + varlen_sparse_feature_columns}
        # )
        # 즉 linear 처리일때는 1 아니면 output emb_dim 으로 처리.... ㅅㅍ...

        # input_sparse_feat = torch.concat(sparse_w_x_lxist_, dim=-1)
        #
        # input_dense_feat = dense_feat.matmul(self.dense_weight)
        #
        # input_feat = torch.concat(input_sparse_feat, input_dense_feat, dim=1)
        # fm_term = self.fm_linear(input_feat) + self.pfm(input_feat)
        #
        # dnn_term = self.dnn(input_feat)
        #
        # term = fm_term + dnn_term
        y_pred = torch.sigmoid(fm_term)
        return y_pred

    def get_sparse_w_x_list(self, x):
        w_x_list = []
        begin = 0
        for field_size, emb in zip(self.sparse_feat_cardinality_list, self.sparse_emb_list):
            x_i = x[:, begin:begin + field_size].to(torch.long)  # 500
            w_i_x_i = emb(x_i)
            # sum_w_i_x_i = torch.sum(w_i_x_i, dim=1, keepdim=False)
            # w_x_list.append(sum_w_i_x_i)

            w_x_list.append(w_i_x_i)  # [B, V, E] => [B, E]
            begin = begin + field_size + 1

        return w_x_list


class FactorizationMachine(nn.Module):

    def __init__(self,
                 sparse_feat_cardinality_list: list[int],
                 n_dense_feat: int,
                 emb_size: int):
        super(FactorizationMachine, self).__init__()

        self.n_sparse_grp_feat = len(sparse_feat_cardinality_list)
        self.sparse_feat_cardinality_list = sparse_feat_cardinality_list

        self.linear = nn.Linear(self.n_sparse_grp_feat + n_dense_feat, 1)
        self.sparse_emb_list_for_linear = nn.ModuleList([nn.Embedding(field_size, 1)
                                                         for field_size in sparse_feat_cardinality_list])

        self.sparse_emb_list = nn.ModuleList([nn.Embedding(field_size, emb_size)
                                              for field_size in sparse_feat_cardinality_list])
        self.pfm = PairwiseFactorizationMachine()

    def forward(self, sparse_feat, dense_feat):
        """
        x: [batch_size, column_size, embedding_size]
        """
        # TODO 아래 로직은 dnn 과정에서도 사용한다
        # fm > linear에서 linear_w_x + densefeat, pairwise 과정에서는 w_x
        # dnn 과정에서는 w_x ->  w_x + dense_feat을 사용
        # 쪼개서 DeepFactorizationMachine 에서 처리 가능하도록 하자
        linear_w_x_list = []
        w_x_list = []  # for pairwise
        begin = 0
        for field_size, linear_emb, emb in zip(self.sparse_feat_cardinality_list, self.sparse_emb_list_for_linear,
                                               self.sparse_emb_list):
            x_i = sparse_feat[:, begin:begin + field_size].to(torch.long)

            linear_w_x = linear_emb(x_i)
            sum_linear_w_x = torch.sum(linear_w_x,  dim=1, keepdim=False)
            linear_w_x_list.append(sum_linear_w_x)

            w_i_x_i = emb(x_i)
            w_x_list.append(w_i_x_i)  # [B, V, E] => [B, E]
            begin = begin + field_size + 1

        linear_w_x = torch.concat(linear_w_x_list, dim=-1)
        fm_linear_input_x = torch.concat((linear_w_x, dense_feat), dim=-1)
        fm_linear_and_bias = self.linear(fm_linear_input_x)

        w_x = torch.concat(w_x_list, dim=1)
        fm_pairwise = self.pfm(w_x)
        return fm_linear_and_bias + fm_pairwise


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
