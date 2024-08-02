from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        assert(config.embedding_dim % config.num_heads == 0)
        # self.k = nn.Linear(config.embedding_dim, config.embedding_dim)
        # self.q = nn.Linear(config.embedding_dim, config.embedding_dim)
        # self.v = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.qkv = nn.Linear(config.embedding_dim, config.embedding_dim * 3)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        # x.shape is (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, embed_dim = x.shape
        # (b,seq_len,num_heads, head_dim) --> (b, num_heads, seq_len, head_dim)
        # K = self.k(x).view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).permute(0, 2, 1, 3)
        # Q = self.q(x).view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).permute(0, 2, 1, 3)
        # V = self.v(x).view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).permute(0, 2, 1, 3)
        Q, K, V = self.qkv(x).split(embed_dim, dim=-1)
        Q = Q.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).permute(0, 2, 1, 3)
        #attention computation below
        attn = (Q @ K.transpose(-2, -1)) / (embed_dim ** 0.5)
        attn = attn.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))
        # now the shape is (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        attn = attn @ V
        attn = self.attn_drop(attn)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn = self.linear(attn)
        attn = self.resid_drop(attn)
        return attn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.embedding_dim, config.embedding_dim * 4)
        self.fc2 = nn.Linear(config.embedding_dim * 4, config.embedding_dim)
        self.act = F.gelu
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
        

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 50257 # 50257 for gpt2 vocab size
    block_size: int = 1024
    layers: int = 12
    num_heads: int = 8
    embedding_dim: int = 768
    dropout: float = 0.0
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_emb = nn.Embedding(config.block_size, config.embedding_dim)
        # drop layer for token embedding and positional embedding
        self.dropLayer = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.ln = nn.LayerNorm(config.embedding_dim)
        self.linear_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # init the weight
        self.linear_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
        # need to adjust the std_variation before the residual project, according to gpt2 paper
        residual_proj = ['linear.weight', 'fc2.weight']
        for k, v in self.named_parameters():
            if any(k.endswith(layer) for layer in residual_proj):
                torch.nn.init.normal_(v, mean=0.0, std=0.02/math.sqrt(2 * config.layers))
        
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def configure_optimizers(self, config):
        print("Configuring optimizer.............")
        decay_params = [param for param in self.parameters() if param.requires_grad and param.dim() > 1]
        nondecay_params = [param for param in self.parameters() if param.requires_grad and param.dim() == 1]
        
        decay_rate = config.weight_decay
        param_groups = [{'params': decay_params, 'weight_decay': decay_rate},
                        {'params': nondecay_params, 'weight_decay': 0.0}]
        
        device_type = config.device.type
        fused = (device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters)
        optimizer = torch.optim.AdamW(param_groups, lr=config.lr, eps=config.eps, betas=(config.beta1, config.beta2), fused=fused)
        print(f"Optimizer configured! Weight decay: {decay_rate}; Learning rate: {config.lr}; Beta1 = {config.beta1}, Beta2 = {config.beta2}; Fused = {fused}")
        return optimizer
        
    def forward(self, x, target=None):
        # x.shape is (batch_size, seq_len)
        batch_size, seq_len = x.shape
        pos = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.dropLayer(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        x = self.linear_head(x)
        # Of course we can use softmax here. 
        # However, if the parameters are not inited well, there'll be gradient explosion, the model can not converge
        x = F.log_softmax(x, dim=-1) 
        # x = F.softmax(x, dim=-1)
        loss = None
        # x.shape is (batch_size, seq_len, vocab_size)
        if target is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), target.view(-1))
        return x, loss
    
    @torch.no_grad()
    def generate(self, text, max_length):
        # text.shape: (batch_size, seq_len)
        # text is the input tokens
        tokens = text
        with torch.no_grad():
            for _ in range(max_length):
                output, loss = self.forward(tokens)
                # output shape: (batch_size, seq_len, vocab_size)
                # extract the last column
                nxt_token = output[:,-1,:] # shape: (batch_size, vocab_size)
                k_probs, k_index = torch.topk(nxt_token, 50)
                index = torch.multinomial(-k_probs, 1)
                
                xcol = torch.gather(k_index, -1, index) 
                tokens = torch.cat([tokens, xcol], dim=-1)
        return tokens
            