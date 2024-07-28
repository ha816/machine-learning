import numpy as np
import torch
import torch.nn as nn


class DeepFactorizationMachine(nn.Module):
    """Instantiates the DeepFM Network architecture.
    """

    def __init__(self,
                 sparse_feat_grp_size_list: list[int],
                 n_dense_feat: int,
                 emb_size: int,
                 deep_layers: list[int]):
        super(DeepFactorizationMachine, self).__init__()

        self.sparse_feat_grp_size_list = sparse_feat_grp_size_list
        self.emb_size = emb_size

        self.sparse_emb_list = nn.ModuleList([nn.Linear(field_size, emb_size, bias=False)
                                              for field_size in sparse_feat_grp_size_list])
        self.dense_emb = nn.Linear(n_dense_feat, n_dense_feat * emb_size, bias=False)  # [D.F, D.F * E]

        self.fm = FactorizationMachine(sparse_feat_grp_size_list, n_dense_feat)
        self.dnn = DeepNeuralNetwork((len(sparse_feat_grp_size_list) + n_dense_feat) * emb_size, deep_layers)

        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")

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

        fm_term = self.fm(torch.concat((sparse, dense), dim=-1), emb)
        dnn_term = self.dnn(torch.flatten(emb, start_dim=1))  # [B, (S.G.F + D.F) * E]

        y_pred = torch.sigmoid(fm_term + dnn_term)
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


class FactorizationMachine(nn.Module):

    def __init__(self,
                 sparse_feat_grp_size_list: list[int],
                 n_dense_feat: int):
        super(FactorizationMachine, self).__init__()

        self.linear = nn.Linear(np.sum(sparse_feat_grp_size_list) + n_dense_feat, 1)
        self.pfm = PairwiseFactorizationMachine()

    def forward(self, x, emb_x):
        """
        x: [BatchSize, DimSize(SparseFeatSize + DenseFeatSize)]
        emb_x: [BatchSize, FieldSize(SparseFeatGroupSize + DenseFeatSize), EmbeddingSize]
        """
        linear_and_bias = self.linear(x)
        pairwise = self.pfm(emb_x)
        # print(f"FM {fm_linear_and_bias.mean()} / {fm_pairwise.mean()} / {fm_output.mean()}")
        return linear_and_bias + pairwise


class PairwiseFactorizationMachine(nn.Module):
    """
    Factorization Machine models pairwise (order-2) feature interactions **Without linear term and bias**
    pairwise (order-2) feature interactions refer to the interactions  between every possible pair of features in the dataset.
    """

    def __init__(self):
        super(PairwiseFactorizationMachine, self).__init__()

    def forward(self, x):
        """
        x: 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
        Output: 2D tensor with shape: ``(batch_size, 1)``.
        """
        square_of_sum = torch.pow(torch.sum(x, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(x * x, dim=1, keepdim=True)

        cross_term = square_of_sum - sum_of_square  # (batch_size,1,embedding_size)
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)  # (batch_size,1)
        return cross_term


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
