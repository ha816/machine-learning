import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoInt(nn.Module):
    """Instantiates the DeepFM Network architecture.
    """

    def __init__(self,
                 sparse_feat_grp_size_list: list[int],
                 n_dense_feat: int,
                 emb_size: int,
                 deep_layers: list[int],
                 n_att_layer=2,
                 n_multi_att_head=2):
        super(AutoInt, self).__init__()

        self.sparse_feat_grp_size_list = sparse_feat_grp_size_list
        self.emb_size = emb_size

        self.sparse_emb_list = nn.ModuleList([nn.Linear(field_size, emb_size, bias=False)
                                              for field_size in sparse_feat_grp_size_list])
        self.dense_emb = nn.Linear(n_dense_feat, n_dense_feat * emb_size, bias=False)  # [D.F, D.F * E]

        n_sparse_grp_feat = len(sparse_feat_grp_size_list)
        n_sparse_feat = int(np.sum(sparse_feat_grp_size_list))
        n_dim = n_sparse_feat + n_dense_feat

        self.linear = nn.Linear(n_dim, 1)
        self.auto_int_layers = nn.ModuleList([InteractingLayer(emb_size, n_head=n_multi_att_head)
                                              for _ in range(n_att_layer)])

        self.dnn = DeepNeuralNetwork((n_sparse_grp_feat + n_dense_feat) * emb_size, deep_layers)

        dnn_linear_in_feature = deep_layers[-1] + (n_sparse_grp_feat + n_dense_feat)  * emb_size
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False)

    def forward(self, sparse, dense):
        """
        :param sparse: [BatchSize, SparseFeatSize]
        :param dense: [BatchSize, DenseFeatSize]
        :return:
        """
        spare_grp_list = self.get_spare_feat_grp_list(sparse)
        sparse_grp_emb_list = [emb(sparse_grp) for sparse_grp, emb in zip(spare_grp_list, self.sparse_emb_list)]
        sparse_emb = torch.stack(sparse_grp_emb_list, dim=1)  # [B, S.G.F, E]
        dense_emb = self.dense_emb(dense).reshape(dense.shape[0], -1, self.emb_size)  # [B, D.F, E]
        concat_emb = torch.cat((sparse_emb, dense_emb), dim=1)

        att_input = concat_emb
        for layer in self.auto_int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)

        dnn_out = self.dnn(torch.flatten(concat_emb, start_dim=1)) # (n_sparse_grp_feat + n_dense_feat) * emb_size
        dnn_logit = self.dnn_linear(torch.concat([dnn_out, att_output], dim=-1))
        linear_logit = self.linear(torch.concat((sparse, dense), dim=-1))
        y_pred = torch.sigmoid(dnn_logit + linear_logit)
        return y_pred.squeeze(-1)

    def get_spare_feat_grp_list(self, sparse_feat):
        spare_feat_grp_list = []
        begin = 0
        for field_size in self.sparse_feat_grp_size_list:
            end = begin + field_size
            x_i = sparse_feat[:, begin: end]
            assert x_i.shape[1] == field_size

            spare_feat_grp_list.append(x_i)
            begin = end
        return spare_feat_grp_list


class InteractingLayer(nn.Module):
    """
    A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
    """

    def __init__(self, emb_size, n_head=2, scaling=False):
        super(InteractingLayer, self).__init__()

        if emb_size % n_head != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')

        self.emb_size_per_head = emb_size // n_head
        self.scaling = scaling

        self.W_Query = nn.Parameter(torch.Tensor(emb_size, emb_size))
        self.W_key = nn.Parameter(torch.Tensor(emb_size, emb_size))
        self.W_Value = nn.Parameter(torch.Tensor(emb_size, emb_size))
        self.W_Res = nn.Parameter(torch.Tensor(emb_size, emb_size))

        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, x):
        """
        x: Input shape
        - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
        """
        querys = torch.tensordot(x, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(x, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(x, self.W_Value, dims=([-1], [0]))

        querys = torch.stack(torch.split(querys, self.emb_size_per_head, dim=2))
        keys = torch.stack(torch.split(keys, self.emb_size_per_head, dim=2))
        values = torch.stack(torch.split(values, self.emb_size_per_head, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.emb_size_per_head ** 0.5

        normalized_att_scores = F.softmax(inner_product, dim=-1)  # head_num None F F
        result = torch.matmul(normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        result += torch.tensordot(x, self.W_Res, dims=([-1], [0]))  # self.use_res
        result = F.relu(result)
        return result


class DeepNeuralNetwork(nn.Module):

    def __init__(self, emb_size: int, layers: list[int]):
        """
        :param emb_size: [(SparseGroupFeatSize + DenseFeatSize) * EmbeddingSize]
        :param layers: a list contains the num of each hidden layer's units
        """
        super(DeepNeuralNetwork, self).__init__()

        fc_layers = [nn.Linear(emb_size, layers[0]),
                     nn.BatchNorm1d(layers[0], affine=False),
                     nn.Sigmoid()]

        for i in range(1, len(layers)):
            fc_layers.append(nn.Linear(layers[i - 1], layers[i]))
            fc_layers.append(nn.BatchNorm1d(layers[i], affine=False))
            fc_layers.append(nn.Sigmoid())

        self.deep = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        :param x: [BatchSize, EmbeddingSize == (SparseGroupFeatSize + DenseFeatSize) * EmbeddingSize]
        :return: [BatchSize, layers[-1]]
        """
        dense_output = self.deep(x)
        return dense_output
