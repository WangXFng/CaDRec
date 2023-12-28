import torch
from utils import Utils
import torch.nn as nn
import Constants as C
import torch.nn.functional as F


class hGCNEncoder(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()

        self.heads = nn.ModuleList([
            hGCNLayer(d_model, d_model) for _ in range(n_head)
        ])

    def get_non_pad_mask(self, seq):
        """ Get the non-padding positions. """
        return seq.ne(C.PAD).type(torch.float).unsqueeze(-1)

    def forward(self, output, user_output, sparse_norm_adj, event_type):
        slf_attn_mask_subseq = Utils.get_subsequent_mask(event_type)  # M * L * L
        slf_attn_mask_keypad = Utils.get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)  # M x lq x lk
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        outputs = []
        output = output * self.get_non_pad_mask(event_type)
        for head in self.heads:
            output = head(output, user_output, sparse_norm_adj, event_type, slf_attn_mask)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        return outputs.sum(0)


class hGCNLayer(nn.Module):
    def __init__(self, d_model, d_k):
        super(hGCNLayer, self).__init__()

        self.linear = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.linear.weight)

        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_model, d_k, bias=False)
        # self.w_vs = nn.Linear(d_model, n_head, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        # nn.init.xavier_uniform_(self.w_vs.weight)

        self.temperature = d_model ** 0.5
        self.dropout = nn.Dropout(0.1)

    def forward(self, output, user_output, sparse_norm_adj, event_type, slf_attn_mask):

        q, k = self.w_qs(output), self.w_ks(output)
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2)) * slf_attn_mask

        if C.DATASET == 'Foursquare':
            eps = 0.1
        elif C.DATASET == 'lastfm-2k':
            eps = 0.1
        elif C.DATASET == 'douban-book':
            eps = 0.1
        elif C.DATASET == 'Yelp2018':
            eps = 0.5
        else:
            eps = 0

        if C.ABLATION == 'OlyAtten':
            output = torch.matmul(attn, F.elu(self.linear(output)))
        elif C.ABLATION == 'OlyHGCN' or C.ABLATION == 'w/oSA':
            output = torch.matmul(sparse_norm_adj, F.elu(self.linear(output)))
        elif C.ABLATION == 'Addition':
            output = torch.matmul(sparse_norm_adj + attn, F.elu(self.linear(output)))
        else:
            if C.DATASET == 'Foursquare':
                output = torch.matmul(sparse_norm_adj + torch.sign(sparse_norm_adj) * torch.sign(attn) * F.normalize(attn) * eps,
                                              F.elu(self.linear(output)))
            else:
                output = torch.matmul(sparse_norm_adj + torch.sign(sparse_norm_adj) * F.normalize(attn) * eps,
                                              F.elu(self.linear(output)))
        return output



