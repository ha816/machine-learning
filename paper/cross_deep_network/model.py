import numpy as np
import torch
import torch.nn as nn


class CrossDeepNetwork(nn.Module):
    """
    Cross and Deep Network in Deep & Cross Network for Ad Click Predictions
    """

    def __init__(self,
                 sparse_feat_cardinality_list: list[int],
                 n_dense_feat: int,
                 n_cross_layer: int,
                 deep_layers: list[int]):
        super(CrossDeepNetwork, self).__init__()

        self.n_sparse_feat = len(sparse_feat_cardinality_list)

        adjusted_sparse_feat_cardinality_list = [int(6 * pow(x, 0.25)) for x in sparse_feat_cardinality_list]
        n_sparse_feat_cardinality = np.sum(adjusted_sparse_feat_cardinality_list)

        sparse_emb_list = [nn.Embedding(x, y, scale_grad_by_freq=True)
                           for x, y in zip(sparse_feat_cardinality_list, adjusted_sparse_feat_cardinality_list)]
        # sparse column 마다 cardinality *  adjusted_cardinality

        self.sparse_emb_layer = nn.ModuleList(sparse_emb_list)

        n_input_feat: int = n_sparse_feat_cardinality + n_dense_feat
        self.batch_norm = nn.BatchNorm1d(n_input_feat, affine=False)

        self.CrossNet = CrossNetwork(n_input_feat, n_cross_layer)
        self.DeepNet = DeepNetwork(n_input_feat, deep_layers)

        last_layer_feature_num = n_input_feat + deep_layers[-1]  # the dim of feature in last layer
        self.output_layer = nn.Linear(last_layer_feature_num, 1, bias=False)  # nn.linear는 W_logits의 간단한 모델링
        # nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)

    def forward(self, sparse_feat, dense_feat):
        """
        Embedding and Stacking Layer
        """
        w_x_list = []
        for i in range(self.n_sparse_feat):
            x_i = sparse_feat[:, i].to(torch.long)
            wi_xi = self.sparse_emb_layer[i](x_i)
            w_x_list.append(wi_xi)

        sparse_feat_emb = torch.concat(w_x_list, dim=1)
        input_feature = torch.cat((sparse_feat_emb, dense_feat), 1)
        input_feature = self.batch_norm(input_feature)

        out_cross = self.CrossNet(input_feature)
        out_deep = self.DeepNet(input_feature)
        final_feature = torch.cat((out_cross, out_deep), dim=1)

        predicated_ctr = self.output_layer(final_feature).view(-1)
        predicated_ctr = torch.sigmoid(predicated_ctr)
        return predicated_ctr


class CrossNetwork(nn.Module):
    """
    The ops in this module is x_0 * x_l^T * w_l + x_l + b_l for each layer l, and x_0 is the init input of this module
    """

    def __init__(self, n_input_feat, n_layer: int):
        """
        :param n_input_feat: total num of input_feature, including of the embedding feature and dense feature
        :param n_layer: the number of layer in this module expect of init op
        """
        super(CrossNetwork, self).__init__()
        self.n_layer = n_layer + 1  # add the first calculate

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
        n_sample = x.shape[0]
        x_0 = x.reshape(n_sample, -1, 1)  # B D 1

        x_i = x_0  # B D 1
        for i in range(self.n_layer):
            x_i_t = x_i.reshape(n_sample, 1, -1)  # B 1 D
            x0_xi = torch.matmul(x_0, x_i_t)  # B D D

            x0_xi_w = torch.matmul(x0_xi, self.weight_w[i])  # B D D * B D 1
            x0_xi_w = x0_xi_w.reshape(n_sample, -1, 1)  # B D 1

            bias = self.weight_b[i].reshape(-1, 1)  # D
            x_i = x0_xi_w + bias + x_i  # B D D

        return x_i.squeeze(-1)


class DeepNetwork(nn.Module):
    """
    All of the layer in this module are full-connection layers
    """

    def __init__(self, n_input_feat, n_layers: list):
        """
        :param n_input_feat: total num of input_feature, including of the embedding feature and dense feature
        :param n_layers: a list contains the num of each hidden layer's units
        """
        super(DeepNetwork, self).__init__()
        fc_layers = [nn.Linear(n_input_feat, n_layers[0]),
                     nn.BatchNorm1d(n_layers[0], affine=False),
                     nn.ReLU(inplace=True)]

        for i in range(1, len(n_layers)):
            fc_layers.append(nn.Linear(n_layers[i - 1], n_layers[i]))
            fc_layers.append(nn.BatchNorm1d(n_layers[i], affine=False))
            fc_layers.append(nn.ReLU(inplace=True))

        self.deep = nn.Sequential(*fc_layers)

    def forward(self, x):
        dense_output = self.deep(x)
        return dense_output
