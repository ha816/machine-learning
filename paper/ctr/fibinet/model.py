import itertools

import numpy as np
import torch
from torch import nn


class FiBiNET(nn.Module):
    """Instantiates the Feature Importance and Bilinear feature Interaction NETwork architecture.
    """

    def __init__(self,
                 sparse_feat_grp_size_list: list[int],
                 n_dense_feat: int,
                 emb_size: int,
                 layers: list[int],
                 bi_linear_type="interaction"
                 ):
        super(FiBiNET, self).__init__()

        self.sparse_feat_grp_size_list = sparse_feat_grp_size_list
        self.emb_size = emb_size
        self.sparse_emb_list = nn.ModuleList([nn.Linear(field_size, emb_size, bias=False)
                                              for field_size in sparse_feat_grp_size_list])

        n_sparse_grp_feat = len(sparse_feat_grp_size_list)
        n_sparse_feat = int(np.sum(sparse_feat_grp_size_list))
        n_dim = n_sparse_feat + n_dense_feat

        self.linear = nn.Linear(n_dim, 1)
        self.senet = SENETLayer(n_sparse_grp_feat, int(n_sparse_grp_feat * 0.3))
        self.bi_linear = BiLinearInteraction(n_sparse_grp_feat, emb_size, bi_linear_type)
        self.bi_linear_logit = nn.Linear(emb_size, 1, bias=False)

        dnn_emb_size = n_sparse_grp_feat * (n_sparse_grp_feat - 1) * emb_size + n_dense_feat
        self.dnn = DeepNeuralNetwork(emb_size=dnn_emb_size, layers=layers)

    def forward(self, sparse, dense):
        spare_grp_list = self.get_spare_feat_grp_list(sparse)
        sparse_grp_emb_list = [emb(sparse_grp) for sparse_grp, emb in zip(spare_grp_list, self.sparse_emb_list)]
        sparse_emb = torch.stack(sparse_grp_emb_list, dim=1)  # [batch_size, F.G + D.F, embedding_size]

        bi_linear_sparse_emb = self.bi_linear(sparse_emb)
        senet_sparse_emb = self.senet(sparse_emb)
        bi_linear_senet_sparse_emb = self.bi_linear(senet_sparse_emb)

        sparse_emb = torch.cat((bi_linear_sparse_emb, bi_linear_senet_sparse_emb), dim=1)

        dnn_emb = torch.cat((torch.flatten(sparse_emb, start_dim=1), dense), dim=1)
        dnn_logit = self.dnn(dnn_emb)

        linear_logit = self.linear(torch.concat((sparse, dense), dim=-1))
        final_logit = linear_logit + dnn_logit
        y_pred = torch.sigmoid(final_logit)
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


class SENETLayer(nn.Module):
    """SENETLayer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
    """

    def __init__(self, n_dim: int, n_reduced_dim: int):
        """
        n_dim: number of feature.
        n_reduced_dim: dimensionality of the attention network output space.
        """
        super(SENETLayer, self).__init__()
        self.seq = nn.Sequential(nn.Linear(n_dim, n_reduced_dim, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(n_reduced_dim, n_dim, bias=False),
                                 nn.ReLU())

    def forward(self, x):
        """
        x: [BatchSize, n_dim, EmbeddingSize]
        """
        z = torch.mean(x, dim=-1)  # [B, n(n-1)/2] #
        a = self.seq(z)
        return torch.mul(x, torch.unsqueeze(a, dim=2))


class BiLinearInteraction(nn.Module):
    """BilinearInteraction Layer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size, embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size*(filed_size-1)/2, embedding_size)``.
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **embedding_size** : Positive integer, embedding size of sparse features.
        - **bilinear_type** : String, types of bilinear functions used in this layer.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, n_dim, emb_size, bi_linear_type="interaction"):

        super(BiLinearInteraction, self).__init__()
        self.bi_linear_type = bi_linear_type
        self.bi_linears = nn.ModuleList()
        if self.bi_linear_type == "interaction":
            for _, _ in itertools.combinations(range(n_dim), 2):
                self.bi_linears.append(nn.Linear(emb_size, emb_size, bias=False))

        if self.bi_linear_type == "each":
            for _ in range(n_dim):
                self.bi_linears.append(nn.Linear(emb_size, emb_size, bias=False))

        if self.bi_linear_type == "all":
            self.bi_linears.append(nn.Linear(emb_size, emb_size, bias=False))

    def forward(self, x):
        """
        x = (batch_size,filed_size, embedding_size)
        """
        x = torch.split(x, 1, dim=1)

        if self.bi_linear_type == "interaction":
            p = []
            for v, bi_linear in zip(itertools.combinations(x, 2), self.bi_linears):
                p.append(torch.mul(bi_linear(v[0]), v[1]))

        if self.bi_linear_type == "each":
            p = [torch.mul(self.bi_linears[i](x[i]), x[j])
                 for i, j in itertools.combinations(range(len(x)), 2)]

        if self.bi_linear_type == "all":
            p = [torch.mul(self.bi_linears(v_i), v_j)
                 for v_i, v_j in itertools.combinations(x, 2)]

        return torch.cat(p, dim=1)  # [B, n_dim(n_dim-1), emb_size]


class DeepNeuralNetwork(nn.Module):

    def __init__(self, emb_size: int, layers: list[int]):
        """
        :param emb_size: [(SparseGroupFeatSize + DenseFeatSize) * EmbeddingSize]
        :param layers: a list contains the num of each hidden layer's units
        """
        super(DeepNeuralNetwork, self).__init__()  # 21216
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
