import numpy as np
import torch
from torch import nn


class XDeepFactorizationMachine:
    """Instantiates the xDeepFM architecture.

    """

    def __init__(self,
                 sparse_feat_grp_size_list: list[int],
                 n_dense_feat: int,
                 emb_size: int,
                 neural_layers: list[int],
                 cin_layers: list[int],
                 cin_split_half=True):
        super(XDeepFactorizationMachine, self).__init__()

        self.sparse_feat_grp_size_list = sparse_feat_grp_size_list
        self.emb_size = emb_size

        self.sparse_emb_list = nn.ModuleList([nn.Linear(field_size, emb_size, bias=False)
                                              for field_size in sparse_feat_grp_size_list])
        self.dense_emb = nn.Linear(n_dense_feat, n_dense_feat * emb_size, bias=False)  # [D.F, D.F * E]

        n_dim = np.sum(sparse_feat_grp_size_list) + n_dense_feat
        # TODO XDeepFactorizationMachine 구현
        # TODO 아래 3가지 모델 수정
        self.fm = FactorizationMachine(n_dim)
        self.dnn = DeepNeuralNetwork(emb_size, neural_layers)
        self.cin = CompressedInteractionNetwork(n_dim, cin_layers)

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
        emb = torch.concat((sparse_emb, dense_emb), dim=1)  # [B, S.G.F + D.F, E]

        fm_term = self.fm(torch.concat((sparse, dense), dim=-1))
        dnn_term = self.dnn(torch.flatten(emb, start_dim=1))
        cin_term = self.cin(emb)

        y_pred = torch.sigmoid(fm_term + dnn_term + cin_term)
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


class CompressedInteractionNetwork(nn.Module):
    """Compressed Interaction Network used in xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)``
        ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]``
        if ``split_half=True``,else  ``sum(layer_size)`` .
    """

    def __init__(self,
                 n_dim: int,
                 layers=[128, 128],
                 half_split=True):

        super(CompressedInteractionNetwork, self).__init__()
        if layers[-1] != 1:
            layers.append(1)

        self.n_dims = [n_dim]
        self.half_split = half_split
        self.activation = nn.ReLU()

        self.conv1d_list = nn.ModuleList()
        for idx, size in enumerate(layers):
            self.conv1d_list.append(nn.Conv1d(self.n_dims[-1] * n_dim, size, 1))

            if self.half_split:
                if idx != len(self.n_dims) - 1 and size % 2 == 0:
                    raise ValueError("n_dims must be even number except for the last layer when split_half=True")
                self.n_dims.append(size // 2)
            else:
                self.n_dims.append(size)

    def forward(self, x):
        """
        x: [BatchSize, (SparseGroupFeatSize + DenseFeatSize), EmbeddingSize]
        """
        batch_size, n_feat, n_emb = tuple(x.shape)
        hidden_nn_layers = [x]

        result = []
        for idx, (n_dim, conv1d) in enumerate(zip(self.n_dims, self.conv1d_list)):
            last_hidden_layer = hidden_nn_layers[-1]

            out = torch.einsum('bhd,bmd->bhmd', last_hidden_layer, x)  # (batch_size , hi * m, dim)
            out = out.reshape(batch_size, last_hidden_layer.shape[1] * n_feat, n_emb)  # (batch_size , hi, dim)
            out = self.conv1d(out)  # layers의 차원별로 conv1d가 존재
            out = self.activation(out)

            if self.half_split:
                if idx < len(self.n_dims):
                    curr_out_dim = 2 * [n_dim // 2]  # size int 값의 절반값의 [size//2, size//2]
                    next_hidden, direct_connect = torch.split(out, curr_out_dim, 1)
                else:
                    direct_connect = out
                    next_hidden = 0
            else:
                direct_connect = out
                next_hidden = out

            result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = torch.cat(result, dim=1)  # 각 direct_connect를 쌓아서
        result = torch.sum(result, -1)  # 하나로 합쳐서 반환
        return result


class FactorizationMachine(nn.Module):
    """
    Deal with first_order only
    linear_and_bias term <W, X>
    """

    def __init__(self, n_dim: int):
        super(FactorizationMachine, self).__init__()
        self.linear = nn.Linear(n_dim, 1)

    def forward(self, x):
        """
        x: [BatchSize, DimSize == (SparseFeatSize + DenseFeatSize)]
        """
        return self.linear(x)


class DeepNeuralNetwork(nn.Module):

    def __init__(self, emb_size: int, layers: list[int]):
        """
        :param emb_size: [(SparseGroupFeatSize + DenseFeatSize) * EmbeddingSize]
        :param layers: a list contains the num of each hidden layer's units
        """
        super(DeepNeuralNetwork, self).__init__()
        if layers[-1] != 1:
            layers.append(1)

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
        :return: [BatchSize, 1]
        """
        dense_output = self.deep(x)
        return dense_output
