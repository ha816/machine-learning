import numpy as np
import torch
import torch.nn as nn


class CDNet(nn.Module):
    """
    Cross and Deep Network in Deep & Cross Network for Ad Click Predictions
    """

    def __init__(self,
                 emb_index: list,
                 emb_size: list,
                 dense_feature_num: int,
                 cross_layer_num: int,
                 deep_layer: list):
        """
        :param emb_index: a list to show the index of the embedding_feature.
                                [0,1,2] shows that index 0, index 1 and index 2 are different category feature
                                [0,[1,2]] shows that index 0 is a category feature and [1,2] is another category feature
        :param emb_size: a list to show the num of classes for each category feature
        :param dense_feature_num: the dim of dense feature
        :param cross_layer_num: the num of cross layer in CrossNet
        :param deep_layer: a list contains the num of each hidden layer's units
        """
        super(CDNet, self).__init__()
        if len(emb_index) != len(emb_size):
            raise ValueError(
                "embedding_index length is {}, embedding_size lenght is {} and they two must have same length")

        self.emb_index = emb_index
        self.emb_size = emb_size
        # embedding_index = [0, 1, 2, 3, 4, 5, 6, [7, 8, 9]],
        # embedding_size = [13, 3, 1001, 2, 10, 10, 7, 11],

        # For categorical features, we embed the features in dense vectors of dimension of 6 * category cardinality^1/4
        n_emb = list(map(lambda x: int(6 * pow(x, 0.25)), self.emb_size))  # 변경된 carinality [20, 10, 30, ....]
        input_feature_num: int = np.sum(n_emb) + dense_feature_num

        emb_list = []  # 각 feature 별로 생성 8개
        for i in range(len(emb_size)):
            emb_list.append(nn.Embedding(emb_size[i], n_emb[i], scale_grad_by_freq=True))
            # origin_cardinality * modified_cardinality
            # embedding size * vocabulary size(category의 cardinality에 비례하는 수의 가중치)
            # emb_list[0] (age) = [13; cardinality] [6 * (13 ^ (1/4))]

        # 즉 emb_list가 학습해야할 가중치로 설정하네...?

        self.emb_layer = nn.ModuleList(emb_list)
        self.batch_norm = nn.BatchNorm1d(input_feature_num, affine=False)

        self.CrossNet = CrossNet(input_feature_num, cross_layer_num)
        self.DeepNet = DeepNet(input_feature_num, deep_layer)

        last_layer_feature_num = input_feature_num + deep_layer[-1]  # the dim of feature in last layer
        self.output_layer = nn.Linear(last_layer_feature_num, 1)  # 0, 1 classification

    def forward(self, sparse_feat, dense_feat):
        """
        Embedding and Stacking Layer
        """
        n_sample = sparse_feat.shape[0]

        if isinstance(self.emb_index[0], list):
            emb_feat = torch.mean(
                self.emb_layer[0](sparse_feat[:, self.emb_index[0]].to(torch.long)), dim=1)
        else:
            emb_feat = torch.mean(self.emb_layer[0](
                sparse_feat[:, self.emb_index[0]].to(torch.long).reshape(n_sample, 1)), dim=1)

        for i in range(1, len(self.emb_index)):
            if isinstance(self.emb_index[i], list):
                emb_feat = torch.cat((emb_feat, torch.mean(
                    self.emb_layer[i](sparse_feat[:, self.emb_index[i]].to(torch.long)), dim=1)), dim=1)
            else:
                emb_feat = torch.cat((emb_feat, torch.mean(self.emb_layer[i](
                    sparse_feat[:, self.emb_index[i]].to(torch.long).reshape(n_sample, 1)), dim=1)), dim=1)

        #  sparse_feat의 평균으로 dense하게 만들고 embedding_feature로 취급
        input_feature = torch.cat((emb_feat, dense_feat), 1)
        input_feature = self.batch_norm(input_feature)

        out_cross = self.CrossNet(input_feature)
        out_deep = self.DeepNet(input_feature)
        final_feature = torch.cat((out_cross, out_deep), dim=1)

        pctr = self.output_layer(final_feature).view(-1)
        pctr = torch.sigmoid(pctr)
        return pctr


class CrossNet(nn.Module):
    """
    Cross layer part in Cross and Deep Network
    The ops in this module is x_0 * x_l^T * w_l + x_l + b_l for each layer l, and x_0 is the init input of this module
    """

    def __init__(self, n_input_feat, cross_layer: int):
        """
        :param n_input_feat: total num of input_feature, including of the embedding feature and dense feature
        :param cross_layer: the number of layer in this module expect of init op
        """
        super(CrossNet, self).__init__()
        self.n_layer = cross_layer + 1  # add the first calculate

        weight_w = []
        weight_b = []
        batchnorm = []
        for i in range(self.n_layer):
            weight_w.append(nn.Parameter(torch.nn.init.normal_(torch.empty(n_input_feat))))
            weight_b.append(nn.Parameter(torch.nn.init.normal_(torch.empty(n_input_feat))))
            batchnorm.append(nn.BatchNorm1d(n_input_feat, affine=False))

        self.weight_w = nn.ParameterList(weight_w)
        self.weight_b = nn.ParameterList(weight_b)
        self.batchnorm = nn.ModuleList(batchnorm)

    def forward(self, x):
        res = x

        x = x.reshape(x.shape[0], -1, 1)
        for i in range(self.n_layer):
            inner_mat = torch.transpose(res.reshape(res.shape[0], -1, 1))
            term_mat = torch.bmm(x, inner_mat)  # batch matrix-matrix product

            res += torch.matmul(term_mat, self.weight_w[i]) + self.weight_b[i]
            res = self.batchnorm[i](res)

        return res


class DeepNet(nn.Module):
    """
    Deep part of Cross and Deep Network
    All of the layer in this module are full-connection layers
    """

    def __init__(self, input_feature_num, deep_layer: list):
        """
        :param input_feature_num: total num of input_feature, including of the embedding feature and dense feature
        :param deep_layer: a list contains the num of each hidden layer's units
        """
        super(DeepNet, self).__init__()
        fc_layer_list = []
        fc_layer_list.append(nn.Linear(input_feature_num, deep_layer[0]))
        fc_layer_list.append(nn.BatchNorm1d(deep_layer[0], affine=False))
        fc_layer_list.append(nn.ReLU(inplace=True))
        for i in range(1, len(deep_layer)):
            fc_layer_list.append(nn.Linear(deep_layer[i - 1], deep_layer[i]))
            fc_layer_list.append(nn.BatchNorm1d(deep_layer[i], affine=False))
            fc_layer_list.append(nn.ReLU(inplace=True))
        self.deep = nn.Sequential(*fc_layer_list)

    def forward(self, x):
        dense_output = self.deep(x)
        return dense_output
