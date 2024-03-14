import os
import random
from tqdm import tqdm
from collections import defaultdict as ddict
import numpy as np
import pandas as pd
import nltk
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pad_sequence
from models.graph_models import ConvE, RotatE, TransE, DistMult, Null

GRAPH_MODEL_CLASS = {
    'conve': ConvE,
    'rotate': RotatE,
    'transe': TransE,
    'distmult': DistMult,
    'null': Null,
}


def get_num(dataset_path, dataset, mode='entity'):  # mode: {entity, relation}
    return int(open(os.path.join(dataset_path, dataset, mode + '2id.txt'), encoding='utf-8').readline().strip())


def read(configs, dataset_path, dataset, filename):
    file_name = os.path.join(dataset_path, dataset, filename)
    with open(file_name) as file:
        lines = file.read().strip().split('\n')
    n_triples = int(lines[0])
    triples = []
    for line in lines[1:]:
        split = line.split(' ')
        for i in range(3):
            split[i] = int(split[i])
        triples.append(split)
    assert n_triples == len(triples), 'number of triplets is not correct.'
    return triples


def read_file(configs, dataset_path, dataset, filename, mode='desc'):
    id2name = []
    file_name = os.path.join(dataset_path, dataset, filename)
    with open(file_name, encoding='utf-8') as file:
        lines = file.read().strip('\n').split('\n')
    for i in range(1, len(lines)):
        ids, name = lines[i].split('\t')
        if mode == 'desc':
            name = name.split(' ')
            name = ' '.join(name)
        id2name.append(name)
    return id2name


def read_name(configs, dataset_path, dataset):
    ent_name_file = 'entityid2name.txt'
    rel_name_file = 'relationid2name.txt'
    ent_names = read_file(configs, dataset_path, dataset, ent_name_file, 'name')
    rel_names = read_file(configs, dataset_path, dataset, rel_name_file, 'name')
    return ent_names, rel_names


def get_gt(configs, triples):
    tail_gt, head_gt = ddict(list), ddict(list)
    for triple in triples:
        if not configs.is_temporal:
            head, tail, rel = triple
            tail_gt[(head, rel)].append(tail)
            head_gt[(tail, rel + configs.n_rel)].append(head)
        else:
            head, tail, rel, timestamp = triple
            tail_gt[(head, rel, timestamp)].append(tail)
            head_gt[(tail, rel + configs.n_rel, timestamp)].append(head)
    return tail_gt, head_gt


def dataloader_output_to_tensor(output_dict, key, padding_value=None, return_list=False):
    tensor_out = [out[key] for out in output_dict]
    if return_list:
        return tensor_out
    if not isinstance(tensor_out[0], torch.LongTensor) and not isinstance(tensor_out[0], torch.FloatTensor):
        tensor_out = [torch.LongTensor(value) for value in tensor_out]
    if padding_value is None:
        tensor_out = torch.stack(tensor_out, dim=0)
    else:
        tensor_out = pad_sequence(tensor_out, batch_first=True, padding_value=padding_value)
    return tensor_out


def _get_performance(ranks):
    ranks = np.array(ranks, dtype=float)
    out = dict()
    out['mr'] = ranks.mean(axis=0)
    out['mrr'] = (1. / ranks).mean(axis=0)
    out['hit1'] = np.sum(ranks == 1, axis=0) / len(ranks)
    out['hit3'] = np.sum(ranks <= 3, axis=0) / len(ranks)
    out['hit10'] = np.sum(ranks <= 10, axis=0) / len(ranks)
    return out


def get_performance(model, tail_ranks, head_ranks, mode):
    tail_out = _get_performance(tail_ranks)
    head_out = _get_performance(head_ranks)
    mr = np.array([tail_out['mr'], head_out['mr']])
    mrr = np.array([tail_out['mrr'], head_out['mrr']])
    hit1 = np.array([tail_out['hit1'], head_out['hit1']])
    hit3 = np.array([tail_out['hit3'], head_out['hit3']])
    hit10 = np.array([tail_out['hit10'], head_out['hit10']])

    val_mrr = mrr.mean().item()
    model.log('val_mrr_{}'.format(mode), val_mrr)
    perf = {'mrr': mrr, 'mr': mr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    perf = pd.DataFrame(perf, index=['tail ranking', 'head ranking'])
    perf.loc['mean ranking'] = perf.mean(axis=0)
    for hit in ['hit@1', 'hit@3', 'hit@5', 'hit@10']:
        if hit in list(perf.columns):
            perf[hit] = perf[hit].apply(lambda x: '%.2f%%' % (x * 100))
    return perf


# Helper functions from fastai
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


# Implementation from fastai https://github.com/fastai/fastai2/blob/master/fastai2/layers.py#L338
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, e: float = 0.1, reduction='mean'):
        super().__init__()
        self.e, self.reduction = e, reduction

    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # (1-e)* H(q,p) + e*H(u,p)
        return (1 - self.e) * nll + self.e * (loss / c)


def get_loss_fn(configs):
    if configs.label_smoothing == 0:
        return torch.nn.CrossEntropyLoss()
    elif configs.label_smoothing != 0:
        return LabelSmoothingCrossEntropy(configs.label_smoothing)


def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param


def construct_adj(configs, train):
    device = int(configs.gpus.split(',')[0])
    edge_index, edge_type = [], []

    for sub, obj, rel in train:
        edge_index.append((sub, obj))
        edge_type.append(rel)

    # Adding inverse edges
    for sub, obj, rel in train:
        edge_index.append((obj, sub))
        edge_type.append(rel + configs.n_rel)
    # edge_index: 2 * 2E, edge_type: 2E * 1
    edge_index = torch.LongTensor(edge_index).to(device).t()
    edge_type = torch.LongTensor(edge_type).to(device)

    return edge_index, edge_type


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))