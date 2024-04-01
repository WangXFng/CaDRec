from utils import Utils
import torch.nn
import torch.nn as nn
from utils.Utils import *
from Model.hGCN import hGCNEncoder


class Encoder(nn.Module):
    def __init__(
            self,
            num_types, d_model, n_layers, n_head, dropout):
        super().__init__()
        self.d_model = d_model

        self.layer_stack = nn.ModuleList([
            hGCNEncoder(d_model, n_head)
            for _ in range(n_layers)])

    def forward(self, user_id, event_type, enc_output, user_output, adjacent_matrix):
        """ Encode event sequences via masked self-attention. """

        # get individual adj
        adj = torch.zeros((event_type.size(0), event_type.size(1), event_type.size(1)), device='cuda:0')
        for i, e in enumerate(event_type):
            # the slicing operation
            adj[i] = adjacent_matrix[e - 1, :][:, e - 1]
            # performance can be enhanced by adding the element in the diagonal of the normalized adjacency matrix.
            adj[i] += adjacent_matrix[e - 1, e - 1]

        for i, enc_layer in enumerate(self.layer_stack):
            residual = enc_output
            enc_output = enc_layer(enc_output, user_output, adj, event_type)
            if C.DATASET in {'douban-book'}:
                enc_output += residual

        return enc_output.mean(1)


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.dropout = nn.Dropout(0.5)
        self.temperature = 512 ** 0.5
        self.dim = dim

    def forward(self, user_embeddings, embeddings, pop_encoding, evaluation):
        outputs = []
        if C.ABLATION != 'w/oMatcher':
            if not evaluation:
                # Foursquare 0.5 Yelp2018 0.3 Gowalla 0.1 Brightkite 0.3 ml-1M 0.8 lastfm-2k 0.3 douban-book 0.05
                item_encoding = torch.concat([embeddings[1:], pop_encoding[1:] * C.BETA_1], dim=-1)
                out = user_embeddings.matmul(item_encoding.T)
            else:
                item_encoding = embeddings[1:]
                out = user_embeddings[:, :self.dim].matmul(item_encoding.T)

            # out = user_embeddings.matmul(embeddings.T[:,1:])
            out = F.normalize(out, p=2, dim=-1, eps=1e-05)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0).sum(0)
        out = torch.tanh(outputs)
        return out


class Model(nn.Module):
    def __init__(
            self, num_types, d_model=256, n_layers=4, n_head=4, dropout=0.1, device=0):
        super(Model, self).__init__()

        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=C.PAD)  # dding 0
        self.user_emb = nn.Embedding(C.USER_NUMBER, d_model, padding_idx=C.PAD)  # dding 0

        self.encoder = Encoder(
            num_types=num_types, d_model=d_model,
            n_layers=n_layers, n_head=n_head, dropout=dropout)
        self.num_types = num_types

        self.predictor = Predictor(d_model, num_types)

    def forward(self, user_id, event_type, adjacent_matrix, pop_encoding, evaluation=True):

        non_pad_mask = Utils.get_non_pad_mask(event_type)

        # (K M)  event_emb: Embedding
        enc_output = self.event_emb(event_type)
        user_output = self.user_emb(user_id)

        pop_output = pop_encoding[event_type] * non_pad_mask

        if C.ABLATION != 'w/oUSpec' and C.ABLATION != 'w/oDisen':
            enc_output += torch.sign(enc_output)\
                          * F.normalize(user_output.unsqueeze(1), dim=-1) # * torch.sign(user_output.unsqueeze(1)) \

        output = self.encoder(user_id, event_type, enc_output, user_output, adjacent_matrix)

        user_embeddings = torch.concat([output, torch.mean(pop_output, dim=1) * C.BETA_1], dim=-1)

        prediction = self.predictor(user_embeddings, self.event_emb.weight, pop_encoding, evaluation)

        return prediction, user_embeddings, pop_output
