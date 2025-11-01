import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.dim_key = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def self_attention(self, Q, K, V, mask=None):
        att_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_key)
        if mask is not None:
            att_score = att_score.masked_fill(mask == 0, -1e9)
        prob = torch.softmax(att_score, dim=-1)
        output = torch.matmul(prob, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.reshape(batch_size, seq_length, self.num_heads, self.dim_key).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, n_heads, seq_length, dim_key = x.size()
        return x.transpose(1, 2).contiguous().reshape(batch_size, seq_length, self.dim)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.query(Q))
        K = self.split_heads(self.key(K))
        V = self.split_heads(self.value(V))

        att_output = self.self_attention(Q, K, V, mask)
        output = self.out(self.combine_heads(att_output))
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, dim_ff):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(dim, dim_ff)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(dim_ff, dim)

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    def __init__(self, dim, num_heads, dim_ff, dropout):
        super(Encoder, self).__init__()
        self.self_attention = MultiHeadAttention(dim, num_heads)
        self.feedforward = FeedForward(dim, dim_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_heads, dim_ff, dropout):
        super(Decoder, self).__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.feed_forward = FeedForward(dim, dim_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, source_mask, target_mask):
        attn_output = self.self_attn(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output, _ = self.cross_attn(x, enc_output, enc_output, key_padding_mask=None)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, source_vocab, target_vocab, dim, num_heads, num_layers, dim_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(source_vocab, dim)
        self.decoder_embedding = nn.Embedding(target_vocab, dim)
        self.positional_encoding = PositionalEncoding(dim, max_seq_length)
        self.encoder_layer = Encoder(dim, num_heads, dim_ff, dropout)
        self.decoder_layer = Decoder(dim, num_heads, dim_ff, dropout)
        self.output_layer = nn.Linear(dim, target_vocab)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, source, target):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(2)
        seq_length = target.size(1)
        nopeak = torch.tril(torch.ones((1, seq_length, seq_length), device=target.device)).bool()
        target_mask = target_mask & nopeak
        return source_mask, target_mask

    def forward(self, source, target):
        source_mask, target_mask = self.generate_mask(source, target)
        source_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(source)))
        target_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(target)))

        enc_output = self.encoder_layer(source_embedded, source_mask)
        dec_output = self.decoder_layer(target_embedded, enc_output, source_mask, target_mask)

        output = self.output_layer(dec_output)
        return output
