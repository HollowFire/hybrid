import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    DynamicConv
)

class DynamicConvEncoderLayer(nn.Module):

    def __init__(self, embed_dim, conv_dim, num_heads, kernel_size, weight_dropout=0.1,
                 dropout=0.3, input_dropout=0.0, weight_softmax=True,
                 encoder_glu=False, normalize_before=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_dim = conv_dim
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        if encoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None

        self.conv = DynamicConv(self.conv_dim, kernel_size, padding_l=padding_l,
                                    weight_softmax=weight_softmax,
                                    num_heads=num_heads,
                                    weight_dropout=weight_dropout)
        self.linear2 = Linear(self.conv_dim, self.embed_dim)

        self.dropout = dropout
        self.input_dropout = input_dropout
        self.normalize_before = normalize_before
        self.layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):

        residual = x
        x = self.maybe_layer_norm(x, before=True)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv(x)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(x, after=True)

        return x, None

    def maybe_layer_norm(self, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norm(x)
        else:
            return x

    def extra_repr(self):
        return 'dropout={}, input_dropout={}, normalize_before={}'.format(
            self.dropout, self.input_dropout, self.normalize_before)

class DynamicConvDecoderLayer(nn.Module):

    def __init__(self, embed_dim, conv_dim, num_heads, kernel_size, weight_dropout=0.1,
                 dropout=0.3, input_dropout=0.0, weight_softmax=True,
                 encoder_glu=False, normalize_before=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_dim = conv_dim

        if encoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None

        self.conv = DynamicConv(self.conv_dim, kernel_size, padding_l=kernel_size-1,
                                    weight_softmax=weight_softmax,
                                    num_heads=num_heads,
                                    weight_dropout=weight_dropout)
        self.linear2 = Linear(self.conv_dim, self.embed_dim)

        self.dropout = dropout
        self.input_dropout = input_dropout
        self.normalize_before = normalize_before
        self.conv_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, incremental_state=None, prev_conv_state=None, **unused):

        residual = x
        x = self.maybe_layer_norm(x, before=True)
        if prev_conv_state is not None:
            if incremental_state is None:
                incremental_state = {}
            self.conv._set_input_buffer(incremental_state, prev_conv_state)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        x = self.conv(x, incremental_state=incremental_state)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(x, after=True)

        return x, None

    def maybe_layer_norm(self, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.conv_layer_norm(x)
        else:
            return x

    def extra_repr(self):
        return 'dropout={}, input_dropout={}, normalize_before={}'.format(
            self.dropout, self.input_dropout, self.normalize_before)

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m



# device = torch.device("cuda:0")
# x = torch.rand(4, 2, 8).to(device)
# encoder = LightConvDecoderLayer(8, 16, 4, 3).to(device)
# mask = torch.zeros(2, 4).bool().to(device)
# y = encoder(x, {}, None)
# print(y.size())
