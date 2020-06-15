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
from fairseq.modules import (
    AdaptiveSoftmax,
    DynamicConv,
    LayerNorm,
    PositionalEmbedding,
    LightweightConv,
    MultiheadAttention,
)
from my.layers.lstm_layer import LSTMEncoderLayer, LSTMDecoderLayer
from my.layers.lightconv_layer import LightConvEncoderLayer, LightConvDecoderLayer
from my.layers.dynamicconv_layer import DynamicConvEncoderLayer, DynamicConvDecoderLayer
from my.layers.attention_layer import AttentionEncoderLayer, AttentionDecoderLayer
from my.layers.ff_layer import FFLayer

@register_model('hybrid')
class HybridModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--input-dropout', type=float, metavar='D',
                            help='dropout probability of the inputs')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-conv-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads or LightConv/DynamicConv heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-conv-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads or LightConv/DynamicConv heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')

        """LightConv and DynamicConv arguments"""
        parser.add_argument('--encoder-kernel-size-list', type=lambda x: options.eval_str_list(x, int),
                            help='list of kernel size (default: "[3,7,15,31,31,31,31]")')
        parser.add_argument('--decoder-kernel-size-list', type=lambda x: options.eval_str_list(x, int),
                            help='list of kernel size (default: "[3,7,15,31,31,31]")')
        parser.add_argument('--encoder-glu', type=options.eval_bool,
                            help='glu after in proj')
        parser.add_argument('--decoder-glu', type=options.eval_bool,
                            help='glu after in proj')
        parser.add_argument('--encoder-conv-type', default='dynamic', type=str,
                            choices=['dynamic', 'lightweight'],
                            help='type of convolution')
        parser.add_argument('--decoder-conv-type', default='dynamic', type=str,
                            choices=['dynamic', 'lightweight'],
                            help='type of convolution')
        parser.add_argument('--weight-softmax', default=True, type=options.eval_bool)
        parser.add_argument('--weight-dropout', type=float, metavar='D',
                            help='dropout probability for conv weights')


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

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = HybridEncoder(args, src_dict, encoder_embed_tokens)
        decoder = HybridDecoder(args, tgt_dict, decoder_embed_tokens)
        return HybridModel(encoder, decoder)


    def load_weights(self, file_path):
        # file_path stores the weights of each module as a dict
        encoder = self.encoder
        decoder = self.decoder
        weights_dict = torch.load(file_path, map_location='cpu')

        # TODO: load embedding, etc.

        # enc_layers = None
        # dec_layers = None
        # for name, module in self.named_modules():
        #     if name == 'encoder.layers':
        #         enc_layers = module
        #     if name == 'decoder.layers':
        #         dec_layers = module

        # load encoder layers weights
        # enc_layers_dict = {k.replace("encoder.layers.", ''): v for k, v in weights_dict.items() if
        #                    k.startswith("encoder.layers.")}
        # dec_layers_dict = {k.replace("decoder.layers.", ''): v for k, v in weights_dict.items() if
        #                    k.startswith("decoder.layers.")}


        enc_layers = self.encoder.layers
        dec_layers = self.decoder.layers

        for layer in enc_layers:
            if isinstance(layer, LSTMEncoderLayer):
                prefix = "encoder.layers.lstm."
            elif isinstance(layer, LightConvEncoderLayer):
                kernel_size = layer.kernel_size
                prefix = "encoder.layers.lightconv.{}.".format(kernel_size)
            elif isinstance(layer, AttentionEncoderLayer):
                prefix = "encoder.layers.self-attention."
            elif isinstance(layer, FFLayer):
                prefix = "encoder.layers.ff."
            else:
                raise RuntimeError("Wrong Encoder Layer Type!")
            weights = {k.replace(prefix, ''): v for k, v in weights_dict.items() if k.startswith(prefix)}
            layer.load_state_dict(weights)

        for layer in dec_layers:
            if isinstance(layer, LSTMDecoderLayer):
                prefix = "decoder.layers.lstm."
            elif isinstance(layer, LightConvDecoderLayer):
                kernel_size = layer.kernel_size
                prefix = "decoder.layers.lightconv.{}."
            elif isinstance(layer, AttentionDecoderLayer):
                if layer.self_attention is True:
                    prefix = "decoder.layers.self-attention."
                else:
                    prefix = "decoder.layers.attention."
            elif isinstance(layer, FFLayer):
                prefix = "decoder.layers.ff."
            weights = {k.replace(prefix, ''): v for k, v in weights_dict.items() if k.startswith(prefix)}
            layer.load_state_dict(weights)






class HybridEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.dropout = args.embed_dropout

        self.embed_tokens = embed_tokens
        self.embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.padding_idx = embed_tokens.padding_idx

        self.max_source_positions = args.max_source_positions
        self.embed_positions = PositionalEmbedding(
            self.max_source_positions, self.embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.encoder_layers = args.encoder_layers
        self.convlayers = nn.ModuleList([
            DynamicConvEncoderLayer(args.encoder_embed_dim, args.encoder_embed_dim,
                                    args.encoder_attention_heads, args.encoder_kernel_size_list[i],
                                    input_dropout=args.conv_input_dropout,
                                    weight_dropout=args.conv_weight_dropout,
                                    dropout=args.conv_output_dropout)
            for i in range(self.encoder_layers)
        ])
        self.attnlayers = nn.ModuleList([
            AttentionEncoderLayer(args.encoder_embed_dim, args.encoder_attention_heads, self_attention=True,
                                  attention_dropout=args.attn_weight_dropout,
                                  dropout=args.attn_output_dropout)
            for _ in range(self.encoder_layers)
        ])
        self.fflayers = nn.ModuleList([
            FFLayer(args.encoder_embed_dim, args.encoder_ffn_embed_dim,
                    relu_dropout=args.ff_relu_dropout,
                    dropout=args.ff_output_dropout)
            for _ in range(self.encoder_layers)
        ])
        self.ratios = nn.Parameter(torch.FloatTensor(self.encoder_layers, 1), requires_grad=True)
        self.ratios.data.fill_(0.5)
        # self.ratios = [nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda() for _ in range(7)]
        # for ratio in self.ratios:
        #     ratio.data.fill_(0.5)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(self.embed_dim)

    def forward(self, src_tokens, **unused):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        ### I want to keep the mask anyway
        # if not encoder_padding_mask.any():
        #     encoder_padding_mask = None
        encoder_states = []
        for i in range(self.encoder_layers):
            x1, state1 = self.convlayers[i](x, encoder_padding_mask=encoder_padding_mask)
            x2, state2 = self.attnlayers[i](x, encoder_padding_mask=encoder_padding_mask)
            if state1 is not None:
                encoder_states.append(state1)
            if state2 is not None:
                encoder_states.append(state2)
            x = x1 * self.ratios[i] + x2 * (1 - self.ratios[i])
            # x = 0.5*x1 + 0.5*x2
            x, _ = self.fflayers[i](x, encoder_padding_mask=encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_x': x,
            'encoder_padding_mask': encoder_padding_mask,
            'encoder_lstm_states': self.get_lstm_states(encoder_states)
        }


    def construct_encoder_layer(self, gene):
        if gene['type'] == 'recurrent':
            return LSTMEncoderLayer(**gene['param'])
        elif gene['type'] == 'lightconv':
            return LightConvEncoderLayer(**gene['param'])
        elif gene['type'] == 'dynamicconv':
            return DynamicConvEncoderLayer(**gene['param'])
        elif gene['type'] == 'self-attention':
            return AttentionEncoderLayer(**gene['param'], self_attention=True)
        elif gene['type'] == 'ff':
            return FFLayer(**gene['param'])
        else:
            raise NotImplementedError('Unknown Decoder Gene Type!')

    def get_lstm_states(self, encoder_states):
        # only return the state of the topmost lstm layer
        final_state = None
        for state in encoder_states:
            if state is not None and "lstm_hidden_state" in state.keys():
                final_state = state["lstm_hidden_state"]
        return final_state

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_x'] is not None:
            encoder_out['encoder_x'] = \
                encoder_out['encoder_x'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['encoder_lstm_states'] is not None:
            hiddens, cells = encoder_out['encoder_lstm_states']
            hiddens = hiddens.index_select(1, new_order)
            cells = cells.index_select(1, new_order)
            encoder_out['encoder_lstm_states'] = (hiddens, cells)
        return encoder_out

    def max_positions(self):
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())



class HybridDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.dropout = args.embed_dropout

        self.embed_tokens = embed_tokens
        self.embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.padding_idx = embed_tokens.padding_idx

        self.max_target_positions = args.max_target_positions
        self.embed_positions = PositionalEmbedding(
            self.max_target_positions, self.embed_dim, self.padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.decoder_layers = args.decoder_layers
        self.convlayers = nn.ModuleList([
            DynamicConvDecoderLayer(args.decoder_embed_dim, args.decoder_embed_dim,
                                    args.decoder_attention_heads, args.decoder_kernel_size_list[i],
                                    input_dropout=args.conv_input_dropout,
                                    weight_dropout=args.conv_weight_dropout,
                                    dropout=args.conv_output_dropout
                                    )
            for i in range(self.decoder_layers)
        ])
        self.attnlayers = nn.ModuleList([
            AttentionDecoderLayer(args.decoder_embed_dim, args.decoder_attention_heads, self_attention=True,
                                  attention_dropout=args.attn_weight_dropout,
                                  dropout=args.attn_output_dropout)
            for _ in range(self.decoder_layers)
        ])
        self.encattnlayers = nn.ModuleList([
            AttentionDecoderLayer(args.decoder_embed_dim, args.decoder_attention_heads, self_attention=False,
                                  attention_dropout=args.attn_weight_dropout,
                                  dropout=args.attn_output_dropout)
            for _ in range(self.decoder_layers)
        ])
        self.fflayers = nn.ModuleList([
            FFLayer(args.decoder_embed_dim, args.decoder_ffn_embed_dim,
                    relu_dropout=args.ff_relu_dropout,
                    dropout=args.ff_output_dropout)
            for _ in range(self.decoder_layers)
        ])

        self.ratios = nn.Parameter(torch.FloatTensor(self.decoder_layers, 1), requires_grad=True)
        self.ratios.data.fill_(0.5)
        # self.ratios = [nn.Parameter(torch.tensor([0.5])).cuda() for _ in range(6)]
        # self.ratios = [nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda() for _ in range(6)]
        # for ratio in self.ratios:
        #     ratio.data.fill_(0.5)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(self.embed_dim)

        self.embed_out = Linear(self.embed_dim, len(dictionary))

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)
        # attn = None

        inner_states = [x]

        ## states from the encoder
        encoder_x = None
        encoder_padding_mask = None
        encoder_lstm_states = None
        if encoder_out is not None:
            if 'encoder_x' in encoder_out.keys():
                encoder_x = encoder_out['encoder_x']
            if 'encoder_padding_mask' in encoder_out.keys():
                encoder_padding_mask = encoder_out['encoder_padding_mask']
            if 'encoder_lstm_states' in encoder_out.keys():
                encoder_lstm_states = encoder_out['encoder_lstm_states']

        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None

        self_attn_padding_mask = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)


        state_dict = {
            "encoder_out": encoder_x,
            "incremental_state": incremental_state,
            "encoder_padding_mask": encoder_padding_mask,
            # args for lstm layers
            "encoder_lstm_states": encoder_lstm_states,
            # args for attention layers
            "self_attn_mask": self_attn_mask,
            "self_attn_padding_mask": self_attn_padding_mask
        }

        for i in range(self.decoder_layers):
            x1, _ = self.convlayers[i](x, **state_dict)
            x2, _ = self.attnlayers[i](x, **state_dict)
            x = x1 * self.ratios[i] + x2 * (1 - self.ratios[i])
            # x = 0.5*x1 + 0.5*x2
            x, _ = self.encattnlayers[i](x, **state_dict)
            x, _ = self.fflayers[i](x, **state_dict)
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)
        x = self.embed_out(x)

        return  x, {"inner_states": inner_states}



    def construct_decoder_layer(self, gene):
        if gene['type'] == 'recurrent':
            return LSTMDecoderLayer(**gene['param'])
        elif gene['type'] == 'lightconv':
            return LightConvDecoderLayer(**gene['param'])
        elif gene['type'] == 'dynamicconv':
            return DynamicConvDecoderLayer(**gene['param'])
        elif gene['type'] == 'self-attention':
            return AttentionDecoderLayer(**gene['param'], self_attention=True)
        elif gene['type'] == 'attention':
            return AttentionDecoderLayer(**gene['param'], self_attention=False)
        elif gene['type'] == 'ff':
            return FFLayer(**gene['param'])
        else:
            raise NotImplementedError('Unknown Encoder Gene Type!')


    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


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

@register_model_architecture('hybrid', 'hybrid')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
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


@register_model_architecture('hybrid', 'hybrid_iwslt_base')
def hybrid_iwslt_base(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 7)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_kernel_size_list = getattr(args, 'encoder_kernel_size_list', [3, 7, 7, 15, 15, 31, 31])
    args.decoder_kernel_size_list = getattr(args, 'decoder_kernel_size_list', [3, 7, 7, 15, 15, 31])
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.encoder_glu = getattr(args, 'encoder_glu', True)
    args.decoder_glu = getattr(args, 'decoder_glu', True)
    base_architecture(args)


@register_model_architecture('cs', 'hybrid_iwslt_de_en')
def hybrid_iwslt_de_en(args):
    args.embed_dropout = getattr(args, 'embed_dropout', 0.3)

    args.conv_input_dropout = getattr(args, 'conv_input_dropout', 0.)
    args.conv_weight_dropout = getattr(args, 'conv_weight_dropout', 0.1)
    args.conv_output_dropout = getattr(args, 'conv_output_dropout', 0.3)

    args.attn_weight_dropout = getattr(args, 'attn_weight_dropout', 0.1)
    args.attn_output_dropout = getattr(args, 'attn_output_dropout', 0.3)

    args.ff_relu_dropout = getattr(args, 'ff_relu_dropout', 0.)
    args.ff_output_dropout = getattr(args, 'ff_output_dropout', 0.3)

    hybrid_iwslt_base(args)
