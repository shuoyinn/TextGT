

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from layers import TextGATConv
from layers import TextGATConv_mod 
from layers import TextGCNConv 
from layers import TextGINConv
from layers import TransformerLayer 



class Transformer(nn.Module): 
    
    def __init__(self, embedding_matrix, opt): 
        super().__init__() 
        self.opt = opt 
        self.embedding_matrix = embedding_matrix 
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True) 
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None        # POS emb 
        # self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0) if opt.post_dim > 0 else None    # position emb 
        self.post_emb = nn.Embedding(2 * opt.max_position + 2, opt.post_dim, padding_idx=0) if opt.post_dim > 0 else None    # Limit the distance value, position emb 

        self.deprel_emb = nn.Embedding(opt.deprel_size + 1, opt.deprel_dim, padding_idx=0) if opt.deprel_dim > 0 else None    # deprel emb, +1 for self-looping edge whose index is just opt.deprel_size. Thought to use 0, abandon due to 0 has been used for padding. 

        in_dim = opt.embed_dim + opt.pos_dim + opt.post_dim 

        if opt.use_rnn: 
            self.rnn = nn.LSTM(in_dim, opt.rnn_hidden, opt.rnn_layers, batch_first=True, dropout=opt.rnn_dropout if opt.rnn_layers > 1 else 0.0, bidirectional=opt.bidirect) 
            self.rnn_drop = nn.Dropout(opt.rnn_dropout) 
            self.linear_middle = nn.Linear(opt.rnn_hidden * 2 if opt.bidirect else opt.rnn_hidden, opt.hidden_dim) 
        else: 
            self.linear_in = nn.Linear(in_dim, opt.hidden_dim) 

        self.linear_out = nn.Linear(opt.hidden_dim, opt.polarities_dim) 

        self.emb_drop = nn.Dropout(opt.input_dropout) 
        self.ffn_dropout = opt.ffn_dropout 

        self.graph_convs = nn.ModuleList() 
        self.norms = nn.ModuleList() 
        self.transformer_layers1 = nn.ModuleList() 
        self.transformer_layers2 = nn.ModuleList() 
        
        norm_class = None 
        if opt.norm == 'ln': 
            norm_class = nn.LayerNorm 
        elif opt.norm == 'bn': # note here, input is B x L x F, need to reshape for bn accept B x F x L 
            norm_class = nn.BatchNorm1d 
        
        for i in range(opt.num_layers): 
            if opt.graph_conv_type == 'gat-mod': 
                graph_conv = TextGATConv_mod(hidden_dim=opt.hidden_dim, # F = H * C 
                            num_heads=opt.graph_conv_attention_heads, # H 
                            attn_dropout=opt.graph_conv_attn_dropout,
                            ffn_dropout=opt.ffn_dropout, 
                            edge_dim=opt.deprel_dim, 
                            norm=opt.norm) 
            elif opt.graph_conv_type == 'gcn': 
                graph_conv = TextGCNConv(opt.hidden_dim, opt.deprel_dim) 
            elif opt.graph_conv_type == 'gin': 
                graph_conv = TextGINConv(opt.hidden_dim, 
                            dropout_ratio=opt.ffn_dropout, 
                            norm=opt.norm, 
                            edge_dim=opt.deprel_dim) 
            elif opt.graph_conv_type == 'gat':
                graph_conv = TextGATConv(hidden_dim=opt.hidden_dim, # F = H * C 
                            num_heads=opt.graph_conv_attention_heads, # H 
                            attn_dropout=opt.graph_conv_attn_dropout,
                            ffn_dropout=opt.ffn_dropout, 
                            edge_dim=opt.deprel_dim, 
                            norm=opt.norm) 
            self.graph_convs.append(graph_conv) 
            self.norms.append(norm_class(opt.hidden_dim)) 
            self.transformer_layers1.append(TransformerLayer( 
                                    opt.hidden_dim, 
                                    opt.attention_heads, 
                                    attn_dropout_ratio=opt.attn_dropout, 
                                    ffn_dropout_ratio=opt.ffn_dropout, 
                                    norm=opt.norm)) 
            self.transformer_layers2.append(TransformerLayer( 
                                    opt.hidden_dim, 
                                    opt.attention_heads, 
                                    attn_dropout_ratio=opt.attn_dropout, 
                                    ffn_dropout_ratio=opt.ffn_dropout, 
                                    norm=opt.norm)) 

    def forward(self, inputs): 
        tok, asp, pos, head, deprel, post, mask, length, adj = inputs 
        """
        tok, pos, head, post, mask: B x L 
        mask: guide the final words pooling to aspect 
        """ 

        maxlen = torch.max(length).item() 
        tok = tok[:, :maxlen] 
        pos = pos[:, :maxlen] 
        deprel = deprel[:, :maxlen] 
        post = post[:, :maxlen] 
        mask = mask[:, :maxlen] 
        adj = adj[:, :maxlen, :maxlen] 
        src_mask = (tok != 0) # B x L 

        # Thought limiting the relative distance here, deprecated for runtime overheads, thus implementing in data_utils.py 
        # post = torch.where(post > self.opt.max_position, self.opt.max_position, post) 
        # post = torch.where(post < -self.opt.max_position, -self.opt.max_position, post) 

        # embedding 
        word_embs = self.emb(tok) # B x L -> B x L x F1 
        embs = [word_embs] 
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)] # -> B x L x F2 
        embs += [self.post_emb(post)] 
        embs = torch.cat(embs, dim=2) # -> B x L x (F1 + F2 + F3) 
        embs = self.emb_drop(embs) 

        # h0 = c0 = torch.zeros(self.opt.rnn_layers * 2 if self.opt.bidirect else self.opt.rnn_layers, 
            # tok.size(0), self.opt.rnn_hidden) # no need, already implemented in pytorch 
        
        if self.opt.use_rnn: 
            rnn_output, (hn, cn) = self.rnn(embs) 
            rnn_output = self.rnn_drop(rnn_output) 
            h = self.linear_middle(rnn_output) 
        else: 
            h = self.linear_in(embs) 

        # Thought generating adj here, deprecated for runtime overheads, thus implementing in data_utils.py 
        # adj = generate_batch_adj(head, deprel, length, self.opt.deprel_size, self.opt.directed, self.opt.add_self_loop).to(self.opt.device) 

        e = self.deprel_emb(adj) # (B, L, L, Fe) 

        for i in range(self.opt.num_layers): 
            h0 = h 

            # Graph Conv is replaced 
            h = self.transformer_layers1[i](h, src_mask) 

            # Middle layer 
            h = self.norms[i](h) 
            h = h.relu() 
            h = F.dropout(h, self.ffn_dropout, training=self.training) 

            # Transformer Layer 
            h = self.transformer_layers2[i](h, src_mask) 

            # Skip connection or Jumping Knowledge 
            h = h + h0 

        # avg pooling asp feature 
        aspect_words_num = mask.sum(dim=1).unsqueeze(-1) # (B, L) -> (B, 1) 
        # mask = mask.unsqueeze(-1).repeat(1,1,self.opt.hidden_dim) # (B, L, 1) -> (B, L, F) # no need repeat for broadcasting automatically 
        mask = mask.unsqueeze(-1) # (B, L, 1) 
        out = (h * mask).sum(dim=1) / aspect_words_num # (B, F) / (B, 1) 
        out = self.linear_out(out) 

        return out 

