

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from layers import TextGATConv
from layers import TextGATConv_mod 
from layers import TextGCNConv 
from layers import TextGINConv
from layers import TransformerLayer 


class TextGT_BERT(nn.Module): 

    def __init__(self, bert, opt): 
        super().__init__() 
        self.opt = opt 
        self.bert = bert 

        self.deprel_embedding_layer = nn.Embedding(opt.deprel_size + 1, opt.deprel_dim, padding_idx=0) if opt.deprel_dim > 0 else None    # deprel emb, +1 for self-looping edge whose index is just opt.deprel_size. Thought to use 0, abandon due to 0 has been used for padding. 

        in_dim = opt.bert_dim #+ opt.pos_dim #+ opt.post_dim 
        self.linear_in = nn.Linear(in_dim, opt.hidden_dim) 
        # self.linear_out = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim) 
        self.linear_out = nn.Linear(opt.hidden_dim + opt.bert_dim, opt.polarities_dim) 

        # drop out 
        self.bert_drop = nn.Dropout(opt.bert_dropout) 
        self.pooled_drop = nn.Dropout(opt.bert_dropout) 

        self.ffn_dropout = opt.ffn_dropout 

        self.graph_convs = nn.ModuleList() 
        self.norms = nn.ModuleList() 
        self.transformer_layers = nn.ModuleList() 

        norm_class = None 
        if opt.norm == 'ln': 
            norm_class = nn.LayerNorm 
        elif opt.norm == 'bn': 
            norm_class = nn.BatchNorm1d 
        
        for i in range(opt.num_layers): 
            if opt.graph_conv_type == 'eela': 
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
            self.transformer_layers.append(TransformerLayer( 
                                    opt.hidden_dim, 
                                    opt.attention_heads, 
                                    attn_dropout_ratio=opt.attn_dropout, 
                                    ffn_dropout_ratio=opt.ffn_dropout, 
                                    norm=opt.norm)) 

    def forward(self, inputs): 
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs 

        """
        text_bert_indices, bert_segments_ids, attention_mask: B x L used for BERT 
        adj_dep: B x L x L used for graph conv 
        src_mask: B x L used for transformer layer of TextGT 
        aspect_mask: B x L used for average pooling 
        """ 

        outputs = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids) 

        # Representations after the BERT Encoder, Pooling result of the BERT Pooler 
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output 
        
        gcn_inputs = self.bert_drop(sequence_output) 
        pooled_output = self.pooled_drop(pooled_output) 

        h = self.linear_in(gcn_inputs) 

        e = self.deprel_embedding_layer(adj_dep) # (B, L, L, Fe) 

        for i in range(self.opt.num_layers): 
            h0 = h 

            # Graph Conv 
            h = self.graph_convs[i](h, adj_dep, e) 

            # Middle layer 
            h = self.norms[i](h) 
            h = h.relu() 
            h = F.dropout(h, self.ffn_dropout, training=self.training) 
            
            # Transformer Layer 
            h = self.transformer_layers[i](h, src_mask) 

            # Skip connection or Jumping Knowledge 
            h = h + h0 

        # avg pooling asp feature 
        aspect_words_num = aspect_mask.sum(dim=1).unsqueeze(-1) # (B, L) -> (B, 1) 
        # mask = mask.unsqueeze(-1).repeat(1,1,self.opt.hidden_dim) # (B, L, 1) -> (B, L, F) # no need repeat for broadcasting automatically 
        aspect_mask = aspect_mask.unsqueeze(-1) # (B, L, 1) 
        out = (h * aspect_mask).sum(dim=1) / aspect_words_num # (B, F) / (B, 1) 

        out = torch.cat([out, pooled_output], dim=-1) 

        out = self.linear_out(out) 

        return out 
