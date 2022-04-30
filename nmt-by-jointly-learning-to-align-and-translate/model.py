import torch
from torch import nn, Tensor
from typing import Optional, Tuple

class Attention(nn.Module):
    """An attention module to apply attention mechanism to neural network
    
    Args:
        ec_hidden_size, dc_hidden_size, dtype, device
    
    Inputs:
        annotations: torch.Tensor, (src_len, num_directions=2 * hidden_size) or (src_len, batch_size, num_directions=2 * hidden_size).
        hidden: torch.Tensor, (num_directions=1, hidden_size) or (num_directions=1, batch_size, hidden_size).
    
    Outputs:
        attn: torch.Tensor, (src_len) or (src_len, batch_size).
    """

    def __init__(self, ec_hidden_size: int, dc_hidden_size: int, dtype = None, device = None) -> None:
        factory_kwargs = {'device': device, 'device': device}
        super(Attention, self).__init__()
        self.ec_hidden_size = ec_hidden_size
        self.dc_hidden_size = dc_hidden_size
        self.attn_hidden_size = dc_hidden_size

        self.linear_combined = nn.Linear(dc_hidden_size + 2 * ec_hidden_size, self.attn_hidden_size, **factory_kwargs)
        self.linear_attn = nn.Linear(self.attn_hidden_size, 1, bias=False, **factory_kwargs)

        self._init_weights()

    def _init_weights(self) -> None:
        pass

    def forward(self, annotations: Tensor, hidden: Tensor) -> Tensor:
        assert annotations.dim() in (2, 3) and annotations.size(-1) == 2 * self.ec_hidden_size, "Attention: Invalid annotations shape"
        assert hidden.dim() == annotations.dim() and hidden.size(0) == 1 and hidden.size(-1) == self.dc_hidden_size, "Attention: Invalid hidden shape"

        is_batched = annotations.dim() == 3
        assert not is_batched or annotations.size(1) == hidden.size(1), "Attention: Batch sizes of annotations and hidden don't match"
        
        if not is_batched:
            annotations = annotations.unsqueeze(1)
            hidden = hidden.unsqueeze(1)
        
        # (1, batch_size, dc_hidden_size) -> (src_len, batch_size, dc_hidden_size)
        hidden = hidden.repeat(annotations.size(0), 1, 1)
        combined = torch.cat((hidden, annotations), dim=2)
        combined = torch.tanh(self.linear_combined(combined))

        # (src_len, batch_size, attn_hidden_size) -> (src_len, batch_size)
        attn = self.linear_attn(combined).squeeze(2)
        attn = nn.functional.softmax(attn, dim=1)

        if not is_batched:
            attn = attn.squeeze(1)
        return attn

class BiRNNEncoder(nn.Module):
    """An encoder module with bidirectional RNN for sequence-to-sequence neural network
    
    Args:
        input_size, embed_size, hidden_size, padding_index, dtype, device
    
    Inputs:
        input: torch.Tensor, (src_len) or (src_len, batch_size).
        hidden (optional): torch.Tensor, (num_directions=2, hidden_size) or (num_directions=2, batch_size, hidden_size). Defaults to zero if not provided.
    
    Outputs:
        output: torch.Tensor, (src_len, num_directions=2 * hidden_size) or (src_len, batch_size, num_directions=2 * hidden_size).
        hidden: torch.Tensor, (num_directions=2, hidden_size) or (num_directions=2, batch_size, hidden_size).
    """

    def __init__(self, input_size: int, embed_size: int, hidden_size: int, padding_index: int = 1, dtype = None, device = None) -> None:
        factory_kwargs = {'device': device, 'device': device}
        super(BiRNNEncoder, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_directions = 2 # it uses bidirectional GRU

        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=padding_index, **factory_kwargs)
        self.rnn = nn.GRU(embed_size, hidden_size, bidirectional=True, num_layers=1, **factory_kwargs)

        self._init_weights()

    def _init_weights(self) -> None:
        pass

    def forward(self, input: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        assert input.dim() in (1, 2), "BiRNNEncoder: Invalid input shape"
        assert hidden is None or (hidden.dim() == input.dim() + 1 and hidden.size(0) == self.num_directions and hidden.size(-1) == self.hidden_size), "BiRNNEncoder: Invalid hidden shape"

        is_batched = input.dim() == 2
        assert hidden is None or not is_batched or hidden.size(1) == input.size(1), "BiRNNEncoder: Batch sizes of input and hidden don't match"

        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

class AttentionRNNDecoder(nn.Module):
    """An decoder module with attention mechanism for sequence-to-sequence neural network
    
    Args:
        embed_size, ec_hidden_size, dc_hidden_size, output_size, padding_index, num_maxouts, dtype, device
    
    Inputs:
        input: torch.Tensor, (trg_len) or (trg_len, batch_size) if model is training and otherwise (1) or (1, batch_size) which consists of <SOS> token.
        annotations: torch.Tensor, (src_len, num_directions=2 * hidden_size) or (src_len, batch_size, num_directions=2 * hidden_size).
        hidden (optional): torch.Tensor, (num_directions=1, hidden_size) or (num_directions=1, batch_size, hidden_size). Initialized by a feedforward network with part of annotaions if not provided.
        max_len: a non-negative integer. The maximum length to sample from the decoder. Defaults to 50 if not provided.
        tf_ratio: a float number between 0 and 1. The strength of teacher forcing. Defaults to 0. if not provided, which means it do not use teacher forcing at all.
    
    Outputs:
        preds: torch.Tensor, (trg_len, output_size) or (trg_len, batch_size, output_size) if model is training and otherwise (max_len, output_size) or (max_len, batch_size, output_size).
        hidden: torch.Tensor, (num_directions=1, hidden_size) or (num_directions=1, batch_size, hidden_size).
        attns: torch.Tensor, (trg_len, src_len) or (trg_len, src_len, batch_size) if model is training and otherwise (max_len, src_len) or (max_len, src_len, batch_size).
    """

    def __init__(self, embed_size: int, ec_hidden_size: int, dc_hidden_size: int, output_size: int, padding_index: int = 1, num_maxouts: int = 500, dtype = None, device = None) -> None:
        factory_kwargs = {'device': device, 'device': device}
        super(AttentionRNNDecoder, self).__init__()
        self.embed_size = embed_size
        self.ec_hidden_size = ec_hidden_size
        self.dc_hidden_size = dc_hidden_size
        self.output_size = output_size
        # self.num_maxouts = num_maxouts
        # self.pool_size = 2
        # self.stride = 2
        self.num_directions = 1 # it one-directional GRU

        self.linear_hidden = nn.Linear(ec_hidden_size, dc_hidden_size, **factory_kwargs)
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=padding_index, **factory_kwargs)
        self.attn = Attention(ec_hidden_size, dc_hidden_size, **factory_kwargs)
        self.rnn = nn.GRU(embed_size + 2 * ec_hidden_size, dc_hidden_size, bidirectional=False, num_layers=1, **factory_kwargs)
        # self.linear_maxout = nn.Linear(embed_size + dc_hidden_size + 2 * ec_hidden_size, num_maxouts * self.pool_size, **factory_kwargs)
        # self.linear_pred = nn.Linear(num_maxouts, output_size, **factory_kwargs)
        self.linear_pred = nn.Linear(embed_size + dc_hidden_size + 2 * ec_hidden_size, output_size, **factory_kwargs)

        self._init_weights()

    def _init_weights(self) -> None:
        pass

    def forward(self, input: Tensor, annotations: Tensor, hidden: Optional[Tensor] = None, max_len: int = 50, tf_ratio: float = 0.) -> Tuple[Tensor, Tensor, Tensor]:
        assert input.dim() in (1, 2), "AttentionRNNDecoder: Invalid input shape"
        assert annotations.dim() == input.dim() + 1 and annotations.size(-1) == 2 * self.ec_hidden_size, "AttentionRNNDecoder: Invalid annotations shape"
        assert hidden is None or (hidden.dim() == input.dim() + 1 and hidden.size(0) == 1 and hidden.size(-1) == self.dc_hidden_size), "AttentionRNNDecoder: Invalid hidden shape"
        
        is_batched = input.dim() == 2
        assert not is_batched or annotations.size(1) == input.size(1), "AttentionRNNDecoder: Batch sizes of input and annotations don't match"
        assert hidden is None or not is_batched or hidden.size(1) == input.size(1), "AttentionRNNDecoder: Batch sizes of input and hidden don't match"

        if not is_batched:
            input = input.unsqueeze(1)
            annotations = annotations.unsqueeze(1)
            if hidden is not None: hidden = hidden.unsqueeze(1)
        
        if hidden is None:
            first_backward_hidden = annotations[0, :, self.ec_hidden_size:]
            hidden = self.linear_hidden(first_backward_hidden)
            hidden = torch.tanh(hidden).unsqueeze(0)
        
        if self.training:
            max_len = input.size(0) # if model is training max_len = trg_len
        
        preds = torch.zeros(max_len, input.size(1), self.output_size, dtype=torch.float, device=input.device)
        attns = torch.zeros(max_len, annotations.size(0), input.size(1), dtype=torch.float, device=input.device)

        annotations_var = annotations.permute(1, 0, 2)

        dc_input = input[0].unsqueeze(0) # (1, batch_size)
        for i in range(1, max_len):
            embedded = self.embedding(dc_input)
            attn = self.attn(annotations, hidden)
            attns[i] = attn

            # (batch_size, 1, src_len) x (batch_size, src_len, 2 * ec_hidden_input) -> (batch_size, 1, 2 * ec_hidden_input)
            attn = attn.permute(1, 0).unsqueeze(1)
            context = torch.bmm(attn, annotations_var)
            context = context.permute(1, 0, 2)
            assert context.shape == (1, input.size(1), 2 * self.ec_hidden_size), "AttentionRNNDecoder: Invalid context computation" # To be deleted

            combined = torch.cat((embedded, context), dim=2)
            output, next_hidden = self.rnn(combined, hidden)

            combined = torch.cat((embedded, hidden, context), dim=2)
            # maxout = self.linear_maxout(combined)
            # maxout = nn.functional.max_pool1d(maxout, self.pool_size, self.stride)

            # pred = self.linear_pred(maxout)
            pred = self.linear_pred(combined)
            pred = nn.functional.log_softmax(pred, dim=2)
            # pred (1, batch_size, output_size)
            preds[i] = pred[0]

            # if self.training and torch.rand(1) < tf_ratio:
            if torch.rand(1) < tf_ratio:
                dc_input = input[i].unsqueeze(0)
            else:
                dc_input = pred.argmax(dim=2)
            # dc_input (1, batch_size)

            hidden = next_hidden
        
        if not is_batched:
            preds = preds.squeeze(1)
            hidden = hidden.squeeze(1)
            attns = attns.squeeze(1)
        return preds, hidden, attns

class AttentionRNNNetwork(nn.Module):
    """An network module with attention mechanism for sequence-to-sequence neural network
    
    Args:
        input_size, embed_size, ec_hidden_size, dc_hidden_size, output_size, padding_index, dtype, device
    
    Inputs:
        src: torch.Tensor, (src_len) or (src_len, batch_size).
        trg: torch.Tensor, (trg_len) or (trg_len, batch_size).
        max_len: a non-negative integer. The maximum length to sample from the decoder. Defaults to 50 if not provided.
        tf_ratio: a float number between 0 and 1. The strength of teacher forcing. Defaults to 0. if not provided, which means it do not use teacher forcing at all.
    
    Outputs:
        preds: torch.Tensor, (trg_len, output_size) or (trg_len, batch_size, output_size) if model is training and otherwise (max_len, output_size) or (max_len, batch_size, output_size).
        attns: torch.Tensor, (trg_len, src_len) or (trg_len, src_len, batch_size) if model is training and otherwise (max_len, src_len) or (max_len, src_len, batch_size).
    """

    def __init__(self, input_size: int, embed_size: int, ec_hidden_size: int, dc_hidden_size: int, output_size: int, padding_index: int = 1, num_maxouts: int = 500, dtype = None, device = None) -> None:
        factory_kwargs = {'device': device, 'device': device}
        super(AttentionRNNNetwork, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.ec_hidden_size = ec_hidden_size
        self.dc_hidden_size = dc_hidden_size
        self.output_size = output_size

        self.encoder = BiRNNEncoder(input_size, embed_size, ec_hidden_size, padding_index=padding_index, **factory_kwargs)
        self.decoder = AttentionRNNDecoder(embed_size, ec_hidden_size, dc_hidden_size, output_size, padding_index=padding_index, num_maxouts=num_maxouts, **factory_kwargs)

        self._init_weights()

    def _init_weights(self) -> None:
        pass

    def forward(self, src: Tensor, trg: Tensor, max_len=50, tf_ratio=0.) -> Tuple[Tensor, Tensor, Tensor]:
        assert src.dim() in (1, 2), "AttentionRNNNetwork: Invalid src shape"
        assert trg.dim() in (1, 2), "AttentionRNNNetwork: Invalid trg shape"

        is_batched = src.dim() == 2
        assert src.dim() == trg.dim() and (not is_batched or src.size(1) == trg.size(1)), "AttentionRNNNetwork: Batch sizes of src and trg don't match"

        annotations, _ = self.encoder(src)
        preds, _, attns = self.decoder(trg, annotations, max_len=max_len, tf_ratio=tf_ratio)
        return preds, attns

    def encode(self, src: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        assert src.dim() in (1, 2), "AttentionRNNNetwork: Invalid src shape"
        annotations, hidden = self.encoder(src, hidden)
        return annotations, hidden
    
    def decode(self, trg: Tensor, annotations: Tensor, hidden: Optional[Tensor] = None, max_len=50, tf_ratio=0.) -> Tuple[Tensor, Tensor, Tensor]:
        assert trg.dim() in (1, 2), "AttentionRNNNetwork: Invalid trg shape"
        preds, hidden, attns = self.decoder(trg, annotations, hidden, max_len, tf_ratio)
        return preds, hidden, attns