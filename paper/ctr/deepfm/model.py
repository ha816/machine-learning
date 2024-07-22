# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
import torch
import torch.nn as nn


class DeepFactorizationMachine(nn.Module):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(DeepFactorizationMachine, self).__init__()

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0

        self.fm = FactorizationMachine()

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False).to(device)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.to(device)

    def forward(self, X):
        sparse_emb_list, dense_value_list = self.input_from_feature_columns(X,
                                                                            self.dnn_feature_columns,
                                                                            self.embedding_dict)

        logit = self.linear_model(X)

        fm_input = torch.cat(sparse_emb_list, dim=1)  # Addition
        logit += self.fm(fm_input)
        # spare_embedding을 합쳐 FM 의 입력으로 사.

        dnn_input = combined_dnn_input(
            sparse_emb_list, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred


class FactorizationMachine(nn.Module):
    """
    Factorization Machine models pairwise (order-2) feature interactions
    *Without linear term and bias*
    """

    def __init__(self):
        super(FactorizationMachine, self).__init__()

    def forward(self, x):
        """
        Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
        Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
        """
        # user preferences와 item attirbutes와 같은 성질이 다른 feature를 말함

        # pairwise (order-2) feature interactions refer to the interactions  between every possible pair of features in the dataset.

        # feat x_i and feat x_j
        # <v_i, v_j> denotes dot product of factor vector v_i and v_j
        # x_i x_j the dot product of values of i feat and j feat
        # <v_i, v_j> x_i x_j

        # x_i x_i+1 + ... + x_i+n x_i+n

        # x_i feat emb = x_i x_i+1
        # torch.sum(x, dim=1, keepdim=True) = (batch_size,1,embedding_size)
        # (x_i + x_i+1 + x_i_2 + ....) ^2
        square_of_sum = torch.pow(torch.sum(x, dim=1, keepdim=True), 2)
        # power of 2 of the sum of the embeddings for each feature

        sum_of_square = torch.sum(x * x, dim=1, keepdim=True)
        # (x_i^2 + x_i+1^2 + x_i_2^2 + ....)

        # (x_i + x_i+1 + x_i_2 + ....) ^2 - (x_i^2 + x_i+1^2 + x_i_2^2 + ....)
        # 2x_i x_i+1 + 2x_i x_i+2 + ... 2
        # 2(x_i x_i+1 + x_i x_i+2 + .._ x_i+n x_i_n-1)

        cross_term = square_of_sum - sum_of_square  # (batch_size,1,embedding_size)
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)  # (batch_size,1)
        return cross_term


class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input
