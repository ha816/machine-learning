import numpy as np
from torch import nn


class XDeepFactorizationMachine:
    """Instantiates the xDeepFM architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
    :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
    :param cin_activation: activation function used on feature maps
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: L2 regularizer strength applied to deep net
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
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

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
                 cin_layer_size=(256, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(XDeepFactorizationMachine, self).__init__()

        # 3 가지 모델을 사용
        self.fm = FactorizationMachine()
        self.cin = CompressedInteractionNetwork()
        self.dnn = DeepNeuralNetwork()

        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)  #
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)

            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.cin_layer_size = cin_layer_size  # cin이라는 layer가 새로 생김
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        if self.use_cin:
            field_num = len(self.embedding_dict)  # field_num n_feat_grp
            self.cin = CompressedInteractionNetwork(field_num, cin_layer_size,
                                                    cin_activation, cin_split_half, l2_reg_cin, seed, device=device)

            if cin_split_half:  #
                self.featuremap_num = sum(cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)  # cin_layer_size의 layers의 합이 featmap_num

            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)  #
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                                           l2=l2_reg_cin)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        linear_logit = self.linear_model(X)  # 여전히 fm

        if self.use_cin:
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(final_logit)
        return y_pred


class FactorizationMachine(nn.Module):
    """
    Deal With first_order and second_order.
    linear_and_bias term <W, X>
    second_order term PairwiseFactorizationMachine
    """

    def __init__(self, dim: int):
        super(FactorizationMachine, self).__init__()

        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x: [BatchSize, DimSize == (SparseFeatSize + DenseFeatSize)]
        """
        return self.linear(x)


class CompressedInteractionNetwork(nn.Module):
    """Compressed Interaction Network used in xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)``
        ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]``
        if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **layer_size** : list of int.Feature maps in each layer.
        - **activation** : activation function name used on feature maps.
        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
        - **seed** : A Python integer to use as random seed.
    """

    def __init__(self,
                 field_size: int,
                 layer_size=(128, 128),
                 activation='relu',
                 split_half=True,
                 l2_reg=1e-5,
                 seed=1024):
        super(CompressedInteractionNetwork, self).__init__()

        if len(layer_size) == 0:
            raise ValueError("layer_size must be a list(tuple) of length greater than 1")

        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed

        self.conv1ds = nn.ModuleList()  # conv1d
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        # batch, ?, dim

        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]  #
        final_result = []

        for i, size in enumerate(self.layer_size):  # 매버 layer_size만큼 도네? (tuple)
            # x^(k-1) * x^0
            latest_hidden_layer = hidden_nn_layers[-1]
            # inputs == hidden_nn_layers[0]
            x = torch.einsum('bhd,bmd->bhmd', latest_hidden_layer, hidden_nn_layers[0])  # 가장 최근에
            # x.shape = (batch_size , hi * m, dim)
            x = x.reshape(batch_size, latest_hidden_layer.shape[1] * hidden_nn_layers[0].shape[1], dim)
            # x.shape = (batch_size , hi, dim)
            x = self.conv1ds[i](x)  # layers의 차원별로 conv1d가 존재

            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    # 여기서나 size를 사용함
                    # curr_out
                    curr_out_dim = 2 * [size // 2]  # size int 값의 절반값의 [size//2, size//2]
                    # 뭔가 느낌은 상위 절반은 next_hidden, 나머지 절반은 direct_connect (결과로 활용 하는 임베딩)
                    next_hidden, direct_connect = torch.split(curr_out, curr_out_dim, 1)
                else:
                    # i == len(self.layer_size) - 1
                    # 처음 layer_size를 넘어서는 상황이 오면 next_hideen을 0으로 취급
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)  #
            hidden_nn_layers.append(next_hidden)  # next_hidden을 hidden_nn_layers에 쌓아감 ㅋㅋ 뭐임?

        result = torch.cat(final_result, dim=1)  # 각 direct_connect를 쌓아서
        result = torch.sum(result, -1)  # 하나로 합쳐서 반환
        return result


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
