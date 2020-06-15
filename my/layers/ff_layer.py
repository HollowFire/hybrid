import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import LayerNorm

class FFLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout=0.0, dropout=0.3, normalize_before=False):
        super().__init__()
        self.fc1 = Linear(embed_dim, ffn_embed_dim)
        self.fc2 = Linear(ffn_embed_dim, embed_dim)
        self.relu_dropout = relu_dropout
        self.dropout = dropout
        self.normalize_before = normalize_before
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x, **unused):
        residual = x
        x = self.maybe_layer_norm(x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
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





def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

