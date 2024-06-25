

import os
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset


def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']: 
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])   # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']   # label
                pos = list(d['pos'])         # pos_tag 
                head = list(d['head'])       # head
                deprel = list(d['deprel'])   # deprel 
                aspect_post = [aspect['from'], aspect['to']] 

                # relative position: relative distance to the aspect 
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]
                # aspect mask 
                if len(asp) == 0: 
                    continue 
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]
                
                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,\
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data


def build_tokenizer(fnames, max_length, data_file):
    parse = ParseData
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    ''' vocabulary of dataset '''
    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():   
            self._reverse_vocab_dict[i] = w  
    
    def word_to_id(self, word): 
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, id_):   
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]
    
    def has_word(self, word):
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length
    
    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class Tokenizer(object):
    ''' transform text to indices '''
    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int 
        self.pos_int_to_char = pos_int_to_char
    
    @classmethod
    def from_files(cls, fnames, max_length, parse, lower=True):
        corpus = set()
        pos_char_to_int, pos_int_to_char = {}, {}
        for fname in fnames:
            for obj in parse(fname):
                text_raw = obj['text']
                if lower:
                    text_raw = text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw)) 
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower, pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)
    
    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:] 
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc 
        else:
            x[-len(trunc):] = trunc
        return x
    
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words] 
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()  
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length, 
                                      padding=padding, truncating=truncating)
    
    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ") 
        return text.strip().split()

# pre-process to get the adjacency matrix, with edge types being the elements 
def generate_adj(head, deprel, length, maxlen, self_loop_idx, directed=False, add_self_loop=True): 
    adj = np.zeros((maxlen, maxlen), dtype=np.int64) 
    for i in range(length): 
        h = head[i] 
        if h == 0: 
            continue # the root is virtual node whose index is 0, then if some node's head is 0, then no edge constructed 
        h = h - 1 # originally 0 represents root (a virtual node), here remove it, then real nodes starts from 0 instead of 1 
        # adj[h, i] = deprel[i] 
        adj[i, h] = deprel[i] # if directed, need to use adj's transpose instead of adj itself, because the message passes from i to h if i-th row and h-th column by convention, i.e., all the columns in a same row aggregate to one representation 
    
    if not directed: 
        adj = adj + adj.T 

    if add_self_loop: 
        for i in range(length): 
            adj[i, i] = self_loop_idx 
        # add self-loops, use above not below 
        # adj = original_adj + torch.eye(original_adj.size(0), dtype=original_adj.dtype, device=original_adj.device) # B x L x L + L x L, has no influence on degrees, and avoid divided by 0 
    
    return adj 


class SentenceDataset(Dataset): 

    def __init__(self, fname, tokenizer, opt, vocab_help): 

        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab_help
        data = list()
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2} 
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"): 
            text = tokenizer.text_to_sequence(obj['text']) 
            aspect = tokenizer.text_to_sequence(obj['aspect'])  # max_length=10 

            # post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
            # post = tokenizer.pad_sequence(post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post') 

            # set maximum and minimum values of distances 
            post = [min(max(p, -opt.max_position), opt.max_position) for p in obj['post']] 
            post = [p + opt.max_position + 1 for p in post] # from distance to index: -max_position -> 1, and max_position -> 2 * max_position + 1, 0 is kept as the padding index; total size is 2 * max_position + 2
            post = tokenizer.pad_sequence(post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')  

            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
            pos = tokenizer.pad_sequence(pos, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
            deprel = tokenizer.pad_sequence(deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            aspect_mask = tokenizer.pad_sequence(obj['mask'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post') 

            head = tokenizer.pad_sequence(obj['head'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post') 

            length = obj['length'] 
            polarity = polarity_dict[obj['label']] 

            adj = generate_adj(head, deprel, length, opt.max_length, opt.deprel_size, opt.directed, opt.add_self_loop) 
            # degree = np.ones(opt.max_length, dtype=np.int64) 
            # degree[:length] = adj.sum(axis=-1)[:length] 

            data.append({
                'text': text, 
                'aspect': aspect, 
                'post': post,
                'pos': pos,
                'deprel': deprel, 
                'head': head,
                'adj': adj,
                'mask': aspect_mask,
                'length': length, 
                # 'degree': degree, 
                'polarity': polarity
            })

        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

def _load_wordvec(data_path, embed_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        if embed_dim == 200:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
                    continue
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        elif embed_dim == 300:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>': # avoid them
                    continue
                elif tokens[0] == '<unk>':
                    word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
                word = ''.join((tokens[:-300]))
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
        else:
            print("embed_dim error!!!")
            exit()
            
        return word_vec


def build_embedding_matrix(vocab, embed_dim, data_file): # get a subset of the pretrained word vectors
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = './glove/glove.840B.300d.txt' 
        word_vec = _load_wordvec(fname, embed_dim, vocab)
        for i in range(len(vocab)): 
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


# pre-process to get the adjacency matrix, BERT version 
def generate_adj_bert(head, deprel, self_loop_idx, directed=False, add_self_loop=True): 
    l = len(head) 
    adj = np.zeros([l, l], dtype=np.int64) 
    for i in range(l): 
        h = head[i] 
        if h == 0: 
            continue # the root is virtual node whose index is 0, then if some node's head is 0, then no edge constructed 
        h = h - 1 # originally 0 represents root (a virtual node), here remove it, then real 
        # adj[h, i] = deprel[i] 
        adj[i, h] = deprel[i] # if directed, need to use adj's transpose instead of adj itself, because the message passes from i to h if i-th row and h-th column by convention, i.e., all the columns in a same row aggregate to one representation 
    
    if not directed: 
        adj = adj + adj.T 

    if add_self_loop: 
        for i in range(l): 
            adj[i, i] = self_loop_idx 
        # add self-loops, use above not below 
        # adj = original_adj + torch.eye(original_adj.size(0), dtype=original_adj.dtype, device=original_adj.device) # B x L x L + L x L, has no influence on degrees, and avoid divided by 0 

    return adj 

class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, pos_vocab, dep_vocab, opt): 
        self.data = []
        parse = ParseData
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2} 
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"): 
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list'] 
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ] 

            head = obj['head'] 
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']] 
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']] 

            trunc = pos[:opt.max_length] 
            trunc = np.asarray(trunc, dtype='int64') 
            pos = (np.zeros(opt.max_length) + 0).astype('int64') 
            pos[:len(trunc)] = trunc 

            ori_adj = generate_adj_bert(head, deprel, opt.deprel_size, opt.directed, opt.add_self_loop) 

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)                   # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)          # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)  # for BERT's nested tokenizer, one word can correspond to multiple tokens, i.e., multiple ids. But here are original ids, not the ones in the BERT's dictionary 
            offset = len(left) 
            for ori_i, w in enumerate(term):        
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term) 
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i+offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3: # 2 for additionall concatenating apspect term; 3 for 2 times of sep and 1 cls 
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0) 
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()
                    
            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')
            for i in range(truncate_tok_len):
                for j in range(truncate_tok_len):
                    tok_adj[i][j] = ori_adj[tok2ori_map[i]][tok2ori_map[j]] # getting the subword syntax tree based on the word one 

            context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
                bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1) # opt.max_length == tokenizer.max_seq_len 
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + paddings
            context_asp_ids += paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            # pad adj
            # context_asp_adj_matrix = np.zeros(
                # (tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32') 
            context_asp_adj_matrix = np.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len), dtype=np.int64) 
            context_asp_adj_matrix[1:context_len + 1, 1:context_len + 1] = tok_adj # the input of BERT is [CLS] + sentence with aspect + [SEP] + aspect + [SEP], while the meaningful part about the syntax tree is only "sentence with aspect", so the node used in adjacency matrix should be from 1 to the frist [SEP] 
            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'adj_matrix': context_asp_adj_matrix,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
                'pos': pos 
            }
            self.data.append(data) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
