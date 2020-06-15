import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)
class AttentionEncoderLayer(nn.Module):

    def __init__(self, embed_dim, attention_heads, self_attention=True,
                 attention_dropout=0.1, dropout=0.3, normalize_before=False,
                 # activation_fn='relu', activation_dropout=0
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(self.embed_dim, attention_heads,
                                            dropout=attention_dropout, self_attention=self_attention)
        self.attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = dropout
        # self.activation_fn activation_fn
        # self.activation_dropout = activation_dropout
        self.normalize_before = normalize_before

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        residual = x
        x = self.maybe_layer_norm(x, before=True)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), -1e8)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(x, after=True)

        return x, None

    def maybe_layer_norm(self, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.attn_layer_norm(x)
        else:
            return x


class AttentionDecoderLayer(nn.Module):

    def __init__(self, embed_dim, attention_heads, self_attention=True,
                 add_bias_kv=False, add_zero_attn=False, attention_dropout=0.1,
                 dropout=0.3, normalize_before=False,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        ) if self_attention else None
        self.dropout = dropout
        self.normalize_before = normalize_before
        self.encoder_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=attention_heads,
            kdim=None, vdim=None,
            dropout=attention_dropout,
            encoder_decoder_attention=True
        ) if not self_attention else None
        self.attn_layer_norm = LayerNorm(self.embed_dim, export=False)

    def forward(self, x,
                encoder_out=None,
                incremental_state=None,
                encoder_padding_mask=None,
                prev_self_attn_state=None,
                prev_attn_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False,
                **unused
                ):
        # encoder_out = None
        # encoder_padding_mask = None
        # if encoder_outs is not None:
        #     if 'encoder_out' in encoder_outs.keys():
        #         encoder_out = encoder_outs['encoder_out']
        #     if 'encoder_padding_mask' in encoder_outs.keys():
        #         encoder_padding_mask = encoder_outs['encoder_padding_mask']

        residual = x
        x = self.maybe_layer_norm(x, before=True)

        if self.self_attn is not None:
            if prev_self_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_self_attn_state[:2]
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                if len(prev_self_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
                self.self_attn._set_input_buffer(incremental_state, saved_state)

            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask
            )

        if self.encoder_attn is not None:
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state[:2]
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, _ = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=False,
                need_head_weights=False
            )


        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(x, after=True)

        return x, None

    def maybe_layer_norm(self, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.attn_layer_norm(x)
        else:
            return x

# import torch
# device = torch.device("cuda:0")
# x = torch.rand(4, 2, 8).to(device)
# encoder = AttentionDecoderLayer(8, 4).to(device)
# mask = torch.zeros(2, 4).bool().to(device)
# y = encoder(x, incremental_state=None)
# print(y.size())