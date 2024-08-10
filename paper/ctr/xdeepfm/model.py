import numpy as np
import torch
from torch import nn


class XDeepFactorizationMachine(nn.Module):
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
                                              for field_size in sparse_feat_grp_size_list])  # 모델
        self.dense_emb = nn.Linear(n_dense_feat, n_dense_feat * emb_size, bias=False)  # [D.F, D.F * E] 가중치 모델

        n_sparse_grp_feat = len(sparse_feat_grp_size_list)
        n_sparse_feat = np.sum(sparse_feat_grp_size_list)
        n_dim = n_sparse_feat + n_dense_feat
        self.fm = FactorizationMachine(n_dim)
        self.dnn = DeepNeuralNetwork((n_sparse_grp_feat + n_dense_feat) * emb_size, neural_layers)  #
        self.cin = CompressedInteractionNetwork(n_sparse_grp_feat, cin_layers, cin_split_half)

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
        cin_term = self.cin(sparse_emb)

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

    def __init__(self,
                 n_dim: int,
                 layers: list[int],
                 half_split=True):

        super(CompressedInteractionNetwork, self).__init__()

        for layer in layers[:-1]:
            if layer % 2 != 0:
                raise ValueError("n_dims must be even number except for the last layer when split_half=True")

        self.n_dim = n_dim
        self.half_split = half_split
        self.activation = nn.ReLU()
        self.layers = layers

        self.conv1d_list = nn.ModuleList()
        n_feat_map = 0
        prev_layer = n_dim
        for idx, layer in enumerate(layers):
            self.conv1d_list.append(nn.Conv1d(prev_layer * n_dim, layer, 1))
            prev_layer = layer // 2 if half_split else layer
            n_feat_map += prev_layer

        self.logit = nn.Linear(n_feat_map, 1, bias=False)

    def forward(self, x):
        """
        x: [BatchSize, SparseFeatSize, EmbeddingSize]
        output: [batch_size, 1]
        """
        batch_size, n_feat, n_emb = tuple(x.shape)
        hidden_layers = [x]

        prev_out = x
        result = []
        for idx, conv1d in enumerate(self.conv1d_list):
            # last_hidden_layer = hidden_layers[-1]

            out = torch.einsum('bhd,bmd->bhmd', prev_out, x)
            # out = torch.einsum('bhd,bmd->bhmd', last_hidden_layer, x)  # (batch_size , n_feat, n_feat, n_emb)
            out = out.reshape(batch_size, -1, n_emb)  # (batch_size , n_feat^2 , n_emb)
            # out = out.reshape(batch_size, last_hidden_layer.shape[1] * n_feat, n_emb)  # (batch_size , n_feat^2 , n_emb)
            out = conv1d(out)  # (batch_size, layer, emb)
            out = self.activation(out)

            if self.half_split:
                if out.shape[1] % 2 == 0:
                    split_size = 2 * [out.shape[1] // 2]  # size int 값의 절반값의 [size//2, size//2]
                    # split_size = 2
                    next_hidden, direct_connect = torch.split(out, split_size, 1)  # out을 dim1으로 해서 두 개로 쪼갠다
                else:
                    result.append(out)
                    break
            else:
                direct_connect = out
                next_hidden = out

            result.append(direct_connect)
            prev_out = next_hidden

        feat_map = torch.cat(result, dim=1)
        feat_map = torch.sum(feat_map, -1)
        return self.logit(feat_map)


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
