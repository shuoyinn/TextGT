

import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from typing import Any 
from torch import Tensor 



class TransformerLayer(nn.Module): 
    def __init__(self, hidden_dim, attention_heads, attn_dropout_ratio, ffn_dropout_ratio, norm='ln'): 
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.attention_heads = attention_heads 
        self.attn_dropout = nn.Dropout(attn_dropout_ratio) 

        self.linear_q = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_k = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_v = nn.Linear(hidden_dim, hidden_dim) 

        self.linear_attn_out = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
            ) 

        norm_class = None 
        if norm == 'ln': 
            norm_class = nn.LayerNorm 
        elif norm == 'bn': 
            norm_class = nn.BatchNorm1d 
        self.norm1 = norm_class(hidden_dim) 

        self.norm2 = norm_class(hidden_dim) 
        
        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, 2*hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(ffn_dropout_ratio), 
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
        ) 
    
    def forward(self, x, src_mask): # x shape (B x L x F) 
        q = self.linear_q(x) 
        k = self.linear_k(x) 
        v = self.linear_v(x) 

        dim_split = self.hidden_dim // self.attention_heads 
        q_heads = torch.cat(q.split(dim_split, 2), dim=0) 
        k_heads = torch.cat(k.split(dim_split, 2), dim=0) 
        v_heads = torch.cat(v.split(dim_split, 2), dim=0) 
        
        attention_score = q_heads.bmm(k_heads.transpose(1, 2)) # (B x H, L, L) 

        attention_score = attention_score / math.sqrt(self.hidden_dim // self.attention_heads) 

        inf_mask = (~src_mask).unsqueeze(1).to(dtype=torch.float) * -1e9 # (B, 1, L) 
        inf_mask = torch.cat([inf_mask for _ in range(self.attention_heads)], 0) 
        A = torch.softmax(attention_score + inf_mask, -1) 

        A = self.attn_dropout(A) 
        attn_out = torch.cat((A.bmm(v_heads)).split(q.size(0), 0), 2) # (B, L, F) 

        attn_out = self.linear_attn_out(attn_out) 
        attn_out = attn_out + x 
        attn_out = self.norm1(attn_out) 
        
        out = self.ffn(attn_out) + attn_out 

        out = self.norm2(out) 

        return out 


# for visualization 
class TransformerLayer_v(nn.Module): 
    def __init__(self, hidden_dim, attention_heads, attn_dropout_ratio, ffn_dropout_ratio, norm='ln'): 
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.attention_heads = attention_heads 
        self.attn_dropout = nn.Dropout(attn_dropout_ratio) 

        self.linear_q = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_k = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_v = nn.Linear(hidden_dim, hidden_dim) 

        self.linear_attn_out = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
            ) 

        norm_class = None 
        if norm == 'ln': 
            norm_class = nn.LayerNorm 
        elif norm == 'bn': 
            norm_class = nn.BatchNorm1d 
        self.norm1 = norm_class(hidden_dim) 

        self.norm2 = norm_class(hidden_dim) 
        
        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, 2*hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(ffn_dropout_ratio), 
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
        ) 
    
    def forward(self, x, src_mask): # x shape (B x L x F) 
        q = self.linear_q(x) 
        k = self.linear_k(x) 
        v = self.linear_v(x) 

        dim_split = self.hidden_dim // self.attention_heads 
        q_heads = torch.cat(q.split(dim_split, 2), dim=0) 
        k_heads = torch.cat(k.split(dim_split, 2), dim=0) 
        v_heads = torch.cat(v.split(dim_split, 2), dim=0) 
        
        attention_score = q_heads.bmm(k_heads.transpose(1, 2)) # (B x H, L, L) 

        attention_score = attention_score / math.sqrt(self.hidden_dim // self.attention_heads) 

        inf_mask = (~src_mask).unsqueeze(1).to(dtype=torch.float) * -1e9 # (B, 1, L) 
        inf_mask = torch.cat([inf_mask for _ in range(self.attention_heads)], 0) 
        A = torch.softmax(attention_score + inf_mask, -1) 

        words_attn = A.clone().detach() # (H x B, L, L), if B == 1 then (H, L, L) 

        A = self.attn_dropout(A) 
        attn_out = torch.cat((A.bmm(v_heads)).split(q.size(0), 0), 2) # (B, L, F) 

        attn_out = self.linear_attn_out(attn_out) 
        attn_out = attn_out + x 
        attn_out = self.norm1(attn_out) 
        
        out = self.ffn(attn_out) + attn_out 

        out = self.norm2(out) 

        return out, words_attn 


# PyG's most types of graph convolution derived from MessagePassing are implemented sparsely, while dense ones incompatible with edge features. 

class TextGINConv(nn.Module): 
    def __init__(self, hidden_dim, dropout_ratio, norm='ln', edge_dim: int = None): 
        super().__init__() 

        self.linear_e = nn.Linear(edge_dim, hidden_dim) 
        norm_class = None 
        if norm == 'ln': 
            norm_class = nn.LayerNorm 
        elif norm == 'bn': 
            norm_class = nn.BatchNorm1d 
        self.ffn = nn.Sequential( 
                        nn.Linear(hidden_dim, 2 * hidden_dim), 
                        nn.ReLU(), 
                        nn.Linear(2 * hidden_dim, hidden_dim)) 
    
    def forward(self, x, adj, e): # x is (B, L, F), adj is (B, L, L), e is (B, L, L, Fe) 
        e = self.linear_e(e) # (B, L, L, F) 
        m = (x.unsqueeze(1) + e).relu() # (B, L, L, F) 
        z = torch.zeros_like(m) 
        m = torch.where((adj != 0).unsqueeze(-1), m, z) # (B, L, L, F) 
        out = m.sum(dim=-2) # (B, L, F) 
        out = self.ffn(out + x) 

        # h = torch.bmm(adj, x) # We do not use matrix multiplication which is hard to consider edge features 
        # h = self.ffn(h) 
        # out = self.norm(h + x) 

        return out 


class TextGCNConv(nn.Module): 
    def __init__(self, hidden_dim, edge_dim: int = None): 
        super().__init__() 
        self.linear_x = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_e = nn.Linear(edge_dim, hidden_dim) 
    
    def gcn_norm(self, adj): # original_adj from B x L x L 
        di = torch.sum(adj, dim=2).unsqueeze(-1) # B x L x 1
        dj = torch.sum(adj, dim=1).unsqueeze(1) # B x 1 x L 
        dij = torch.bmm(di.float(), dj.float()) # di * dj 
        dij_inverse_sqrt = dij.float().pow_(-0.5) 
        # dij_inverse_sqrt[dij_inverse_sqrt == float('inf')] = 0 # already
        z = torch.zeros_like(dij_inverse_sqrt) 
        dij_inverse_sqrt = torch.where(adj, dij_inverse_sqrt, z) # filter out 

        return dij_inverse_sqrt 

    def forward(self, x, adj, e): # x is (B, L, F), adj is (B, L, L), e is (B, L, L, Fe) 
        x = self.linear_x(x) # (B, L, F) 
        e = self.linear_e(e) # (B, L, L, F) 
        x = x.unsqueeze(1) # (B, 1, L, F) 
        m = (x + e).relu() # (B, L, L, F) 
        # adj = adj + torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device) # B x L x L + L x L -> B x L x L 
        z = torch.zeros_like(m) 
        m = torch.where((adj != 0).unsqueeze(-1), m, z) # (B, L, L, F) 

        adj = (adj != 0) 
        adj = self.gcn_norm(adj).unsqueeze(-1) # (B, L, L, 1) 

        out = (adj * m).sum(dim=-2) # (B, L, F) 

        # out = torch.bmm(adj, x) # We do not use matrix multiplication which is inconvenient to handle edge features 

        return out 


def glorot(value: Any): 
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


class TextGATConv(nn.Module): 
    def __init__(
         self, 
        hidden_dim, # F = H * C 
        num_heads: int = 1, # H 
        negative_slope: float = 0.2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0, 
        edge_dim: int = None, 
        norm='ln'): 

        super().__init__() 

        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.negative_slope = negative_slope
        self.attn_dropout = attn_dropout 
        self.edge_dim = edge_dim

        self.linear_x = nn.Linear(hidden_dim, hidden_dim,
                                  bias=False, weight_initializer='glorot') 
        self.att_src = nn.Parameter(torch.Tensor(num_heads, hidden_dim // num_heads)) 
        self.att_dst = nn.Parameter(torch.Tensor(num_heads, hidden_dim // num_heads)) 
        self.linear_e = nn.Linear(edge_dim, hidden_dim, bias=False,
                                weight_initializer='glorot') 
        self.att_e = nn.Parameter(torch.Tensor(num_heads, hidden_dim // num_heads)) 
        glorot(self.att_src) 
        glorot(self.att_dst) 
        glorot(self.att_e) 

        norm_class = None 
        if norm == 'ln': 
            norm_class = nn.LayerNorm
        elif norm == 'bn': 
            norm_class = nn.BatchNorm1d 

        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            norm_class(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(ffn_dropout) 
        ) 
    
    def attention_matrix(self, x, adj, e): 
        # B, L, H, C = x.size(0), x.size(1), x.size(2), x.size(3) 
        xj = (x * self.att_src).sum(dim=-1) # (B, L, H, C) * (H, C) then sum -> (B, L, H) 
        xi = (x * self.att_dst).sum(dim=-1) # (B, L, H, C) * (H, C) then sum -> (B, L, H) 
        e = (e * self.att_e).sum(dim=-1) # (B, L, L, H, C) * (H, C) then sum -> (B, L, L, H) 
        xj = xj.unsqueeze(dim=1) # (B, 1, L, H) 
        xi = xi.unsqueeze(dim=2) # (B, L, 1, H) 
        # xj = xj.expand(B, L, L, H) # no need, auto broadcast 
        # xi = xi.expand(B, L, L, H) # no need, auto broadcast 
        xij = xi + xj + e 
        xij = F.leaky_relu(xij, self.negative_slope) # (B, L, L, H) 
        
        inf = torch.ones_like(xij) * -1e9 
        alpha = torch.where((adj != 0).unsqueeze(-1), xij, inf) # (B, L, L, H) 

        alpha = torch.softmax(alpha, dim=2) 
        alpha = alpha.permute(0, 3, 1, 2) # (B, H, L, L) 
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training) 
        
        return alpha 

    def forward(self, x, adj, e): # x is (B, L, F), adj is (B, L, L), e is (B, L, L, Fe) 
        B, L = x.size(0), x.size(1) 
        H, C = self.num_heads, self.hidden_dim // self.num_heads 
        x = self.linear_x(x).view(B, L, H, C) 
        e = self.linear_e(e).view(B, L, L, H, C) 
        alpha = self.attention_matrix(x, adj, e) 
        x = x.transpose(1, 2) # (B, H, L, C) 
        out = torch.bmm(alpha.contiguous().view(-1, L, L), x.contiguous().view(-1, L, C)).view(B, H, L, C) # (B, H, L, L), (B, H, L, C) -> (B, H, L, C) 
        out = out.transpose(1, 2) # (B, L, H, C) 

        out = out.contiguous().view(B, L, H * C)
        out = self.ffn(out) 

        return out 


class TextGATConv_mod(nn.Module): # little try to modify GAT 
    def __init__(
        self, 
        hidden_dim, # F = H * C 
        num_heads: int = 1, # H 
        negative_slope: float = 0.2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0, 
        edge_dim: int = None, 
        norm='ln'): 

        super().__init__() 

        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.negative_slope = negative_slope 
        self.attn_dropout = attn_dropout 
        self.ffn_dropout = ffn_dropout 
        self.edge_dim = edge_dim 

        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.linear_kv = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.linear_e = nn.Linear(edge_dim, hidden_dim, bias=False) 

        norm_class = None 
        if norm == 'ln': 
            norm_class = nn.LayerNorm
        elif norm == 'bn': 
            norm_class = nn.BatchNorm1d 

        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            norm_class(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(ffn_dropout) 
        ) 
    
    def attention_matrix(self, q, kv, adj): 
        C = q.size(-1) 
        alpha = (q * kv).sum(dim=-1) / math.sqrt(C) # (B, H, L, L, C) -> (B, H, L, L) 
        alpha = F.leaky_relu(alpha, self.negative_slope) # (B, H, L, L) 
        alpha = alpha.permute(0, 2, 3, 1) # (B, L, L, H) 
        
        inf = torch.ones_like(alpha) * -1e9 
        # print(adj.shape) 
        # print(alpha.shape) 
        # print(inf.shape) 
        alpha = torch.where((adj != 0).unsqueeze(-1), alpha, inf) # (B, L, L, H) 
        # alpha = torch.where(adj != 0, alpha, inf) # (B, L, L, H) 
        alpha = alpha.permute(0, 3, 1, 2) # (B, H, L, L) 

        alpha = torch.softmax(alpha, dim=-1) 
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training) 
        
        return alpha 

    def forward(self, x, adj, e): # x is (B, L, F), adj is (B, L, L), e is (B, L, L, Fe) 
        B, L = x.size(0), x.size(1) 
        H, C = self.num_heads, self.hidden_dim // self.num_heads 
        q = self.linear_q(x).view(B, L, H, C).transpose(1, 2) # (B, H, L, C) 
        kv = self.linear_kv(x).view(B, L, H, C).transpose(1, 2) # (B, H, L, C) 
        e = self.linear_e(e).view(B, L, L, H, C).permute(0, 3, 1, 2, 4) # (B, H, L, L, C) 
        # q (k + e) == qk + qe，qk ...  no, directly broadcast 
        q = q.unsqueeze(3) # (B, H, L, 1, C) 
        kv = kv.unsqueeze(2) # (B, H, 1, L, C) 
        kv = kv + e # (B, H, L, L, C) 
        alpha = self.attention_matrix(q, kv, adj) # (B, H, L, L) 
        alpha = alpha.unsqueeze(-1) # (B, H, L, L, 1) 
        out = (alpha * kv).sum(dim=-2) # (B, H, L, L, C) -> (B, H, L, C) 
        out = out.permute(0, 2, 1, 3) # (B, L, H, C) 
        # print(out.shape) 
        out = out.contiguous().view(B, L, H * C) # (B, L, H * C) 
        out = self.ffn(out) 

        return out 


