import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Atencao traduzida para PyTorch
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output            = torch.matmul(attention_weights, V)

        return output, attention_weights

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=128, d_ff=512):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)

        # ReLU
        x = F.relu(x)
        x = self.linear2(x)
        
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, d_ff=512):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.ffn       = FeedForwardNetwork(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attention_output, _  = self.attention(Q, K, V, mask)
        x                    = self.norm1(x + attention_output)

        ffn_output = self.ffn(x)
        x          = self.norm2(x + ffn_output)

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model=128, d_ff=512):
        super().__init__()

        self.W_q1 = nn.Linear(d_model, d_model)
        self.W_k1 = nn.Linear(d_model, d_model)
        self.W_v1 = nn.Linear(d_model, d_model)

        self.W_q2 = nn.Linear(d_model, d_model)
        self.W_k2 = nn.Linear(d_model, d_model)
        self.W_v2 = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.ffn       = FeedForwardNetwork(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, y, Z, target_mask=None):
        # Masked self-attention
        Q1 = self.W_q1(y)
        K1 = self.W_k1(y)
        V1 = self.W_v1(y)

        self_attention_output, _  = self.attention(Q1, K1, V1, target_mask)
        y_norm1                   = self.norm1(y + self_attention_output)

        # Cross-attention
        Q2 = self.W_q2(y_norm1)
        K2 = self.W_k2(Z)
        V2 = self.W_v2(Z)

        cross_attention_output, _  = self.attention(Q2, K2, V2)
        y_norm2                    = self.norm2(y_norm1 + cross_attention_output)

        # Feed Forward Network
        ffn_output  = self.ffn(y_norm2)
        y_norm3     = self.norm3(y_norm2 + ffn_output)

        return y_norm3

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, d_ff=512):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.encoder = EncoderLayer(d_model, d_ff)
        self.decoder = DecoderLayer(d_model, d_ff)

        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, target_mask=None):

        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)

        Z              = self.encoder(src_emb)
        decoder_output = self.decoder(tgt_emb, Z, target_mask)

        logits = self.fc_out(decoder_output)

        return logits
