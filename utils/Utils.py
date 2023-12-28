import torch
import Constants as C
import torch.nn.functional as F


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(C.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(C.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def type_loss(prediction, label, event_time, test_label, opt):
    """ Event prediction loss, cross entropy or label smoothing. """

    prediction = torch.squeeze(prediction[:, :], 1)

    multi_hots = torch.zeros(label.size(0), C.ITEM_NUMBER, device='cuda:0', dtype=torch.float32)

    for i, (t, ti, tl) in enumerate(zip(label, event_time, test_label)):
        multi_hots[i][t[t!=0]-1], multi_hots[i][tl[tl!=0]-1] = opt.beta, opt.lambda_

    log_prb = F.logsigmoid(prediction)
    multi_hots = multi_hots * (1 - opt.smooth) + (1 - multi_hots) * opt.smooth / C.ITEM_NUMBER
    predict_loss = -(multi_hots * log_prb)

    loss = torch.sum(predict_loss)

    return loss


def l2_reg_loss(reg, model, event_type):
    emb_loss = 0
    for e in event_type:
        r = model.event_emb(e[e != 0])
        emb_loss += torch.norm(r, p=2).sum() * reg

    return emb_loss / len(event_type)


def popularity_loss(popularity_pred, in_degree):
    return (popularity_pred - in_degree) ** 2

