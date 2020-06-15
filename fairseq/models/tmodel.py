import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)


@register_model('tmodel')
class TModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        return

    @classmethod
    def build_model(cls, args, task):

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        encoder_embed_tokens = build_embedding(
            src_dict, args.encoder_embed_dim, args.encoder_embed_path
        )
        decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        )

        encoder = TEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TDecoder(args, tgt_dict, decoder_embed_tokens)
        return TModel(encoder, decoder)

class TEncoder(FairseqEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)

        self.embed_tokens = embed_tokens
        self.embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.hidden_dim = 256
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim)

    def forward(self, src_tokens, **unused):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x = x.transpose(0, 1)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        x, state = self.lstm(x)
        return {
            'encoder_out': x,
            'encoder_padding_mask': encoder_padding_mask,
            'hidden_state': state
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['hidden_state'] is not None:
            encoder_out['hidden_state'] = \
                (encoder_out['hidden_state'][0].index_select(1, new_order),
                 encoder_out['hidden_state'][1].index_select(1, new_order))
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""

        return self.max_source_positions

class TDecoder(FairseqIncrementalDecoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        self.embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.hidden_dim = 256
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim)

        self.proj_vocab = Linear(self.hidden_dim, len(dictionary))

    def forward(self, prev_output_tokens, encoder_out=None, encoder_hidden_state=None, incremental_state=None, **kwargs):

        hidden_state = None
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

            hidden_state = self._get_hidden_state(incremental_state)
            if hidden_state is None:
                hidden_state = encoder_hidden_state


        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        x = x.transpose(0, 1)

        if hidden_state is not None:
            x, dec_state = self.lstm(x, hidden_state)
        else:
            x, dec_state = self.lstm(x)

        if incremental_state is not None:
            self._set_hidden_state(incremental_state, dec_state)

        x = x.transpose(0, 1)

        x = self.proj_vocab(x)

        return x, {'hidden_state': dec_state}



    def _set_hidden_state(self, incremental_state, hidden_state):
        return utils.set_incremental_state(self, incremental_state, 'hidden_state', hidden_state)

    def _get_hidden_state(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'hidden_state')

    def reorder_incremental_state(self, incremental_state, new_order):
        hidden_state = self._get_hidden_state(incremental_state)
        if hidden_state is not None:
            h = hidden_state[0].index_select(1, new_order)
            c = hidden_state[1].index_select(1, new_order)
            self._set_hidden_state(incremental_state, (h, c))

    def max_positions(self):
        """Maximum output length supported by the decoder."""

        return self.max_target_positions


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

@register_model_architecture('tmodel', 'tmodel_iwslt_de_en')
def tmodel_iwslt_de_en(args):
    base_architecture(args)

@register_model_architecture('tmodel', 'tmodel')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 7)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.encoder_conv_dim = getattr(args, 'encoder_conv_dim', args.encoder_embed_dim)
    args.decoder_conv_dim = getattr(args, 'decoder_conv_dim', args.decoder_embed_dim)

    args.encoder_kernel_size_list = getattr(args, 'encoder_kernel_size_list', [3, 7, 15, 31, 31, 31, 31])
    args.decoder_kernel_size_list = getattr(args, 'decoder_kernel_size_list', [3, 7, 15, 31, 31, 31])
    if len(args.encoder_kernel_size_list) == 1:
        args.encoder_kernel_size_list = args.encoder_kernel_size_list * args.encoder_layers
    if len(args.decoder_kernel_size_list) == 1:
        args.decoder_kernel_size_list = args.decoder_kernel_size_list * args.decoder_layers
    assert len(args.encoder_kernel_size_list) == args.encoder_layers, "encoder_kernel_size_list doesn't match encoder_layers"
    assert len(args.decoder_kernel_size_list) == args.decoder_layers, "decoder_kernel_size_list doesn't match decoder_layers"
    args.encoder_glu = getattr(args, 'encoder_glu', True)
    args.decoder_glu = getattr(args, 'decoder_glu', True)
    args.input_dropout = getattr(args, 'input_dropout', 0.1)
    args.weight_dropout = getattr(args, 'weight_dropout', args.attention_dropout)