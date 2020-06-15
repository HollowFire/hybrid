import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm


class LSTMEncoderLayer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_layers=1, dropout=0.1,
                 bidirectional=False, left_pad=True, normalize_before=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        # self.input_dropout = input_dropout
        self.dropout = dropout

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            bidirectional=bidirectional
        )
        self.left_pad = left_pad

        self.output_units = hidden_dim
        if bidirectional:
            self.output_units *= 2
        self.linear = Linear(self.output_units, self.embed_dim)
        self.layer_norm = LayerNorm(embed_dim)
        self.normalize_before = normalize_before



    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(x, before=True)
        converted_mask = None
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
            if self.left_pad:
                x, converted_mask = convert_padding_direction(x, encoder_padding_mask, left_to_right=True)
        max_len, bsz, _ = x.size()
        src_lengths = torch.bitwise_not(converted_mask).long().sum(dim=1).data.tolist() if \
                        converted_mask is not None else [max_len for _ in range(bsz)]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths)

        if self.bidirectional:
            state_size= 2 * self.num_layers, bsz, self.hidden_dim
        else:
            state_size = self.num_layers, bsz, self.hidden_dim
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=0)
        if encoder_padding_mask is not None:
            if self.left_pad:
                x, _ = convert_padding_direction(x, converted_mask, right_to_left=True)
        assert list(x.size()) == [max_len, bsz, self.output_units]
        x = self.linear(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(x, after=True)

        if self.bidirectional:
            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        return x, {"lstm_hidden_state": (final_hiddens, final_cells)}

    def maybe_layer_norm(self, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norm(x)
        else:
            return x


class LSTMDecoderLayer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_layers=1, dropout=0.1,
                 normalize_before=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_dim + embed_dim if layer == 0 else hidden_dim,
                hidden_size=hidden_dim
            )
            for layer in range(num_layers)
        ])

        self.linear = Linear(hidden_dim, embed_dim)
        self.dropout = dropout
        self.layer_norm = LayerNorm(embed_dim)
        self.normalize_before = normalize_before

    def forward(self, x, incremental_state=None, encoder_lstm_states=None, **unused):

        residual = x
        x = self.maybe_layer_norm(x, before=True)
        seqlen, bsz, _ = x.size()

        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            if encoder_lstm_states is not None:
                encoder_hiddens, encoder_cells = encoder_lstm_states
                prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
                prev_cells = [encoder_cells[i] for i in range(num_layers)]
            else:
                state_size = bsz, self.hidden_dim
                prev_hiddens = [x.new_zeros(*state_size) for _ in range(num_layers)]
                prev_cells = [x.new_zeros(*state_size) for _ in range(num_layers)]
                # prev_hiddens = [x.new_zeros(*state_size) for i in range(num_layers)]
                # prev_cells = [x.new_zeros(*state_size) for i in range(num_layers)]
            input_feed = x.new_zeros(bsz, self.hidden_dim)

        outs = []
        for j in range(seqlen):
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):

                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                # input = F.dropout(hidden, p=self.dropout, training=self.training)
                input = hidden
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = hidden
            input_feed = out
            outs.append(out)

        utils.set_incremental_state(self, incremental_state, 'cached_state',
                                    (prev_hiddens, prev_cells, input_feed))


        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_dim)
        x = self.linear(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + residual
        x = self.maybe_layer_norm(x, after=True)

        return x, None

    def reorder_incremental_state(self, incremental_state, new_order):
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
            prev_hiddens = [hidden.index_select(0, new_order) for hidden in prev_hiddens]
            prev_cells = [cell.index_select(0, new_order) for cell in prev_cells]
            input_feed = input_feed.index_select(0, new_order)
            utils.set_incremental_state(self, incremental_state, 'cached_state',
                                        (prev_hiddens, prev_cells, input_feed))



    def maybe_layer_norm(self, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norm(x)
        else:
            return x

def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m

def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m

def convert_padding_direction(x, pad_mask, right_to_left=False, left_to_right=False):
    # x: T*B*C   pad_mask: B*T
    assert right_to_left ^ left_to_right
    x = x.transpose(0, 1) # B*T
    if not pad_mask.any():
        # no padding, return early
        return x.transpose(0, 1), pad_mask
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return x.transpose(0, 1), pad_mask
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return x.transpose(0, 1), pad_mask
    max_len = x.size(1)
    ## TODO
    range = buffered_arange(max_len).to(torch.long).to(x.device).unsqueeze(-1).expand_as(x)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True).unsqueeze(-1).expand_as(x)

    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)

    x = x.gather(1, index)
    pad_mask = pad_mask.gather(1, index[:,:,0])

    x = x.transpose(0, 1) # T*B*C
    return x, pad_mask



def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

