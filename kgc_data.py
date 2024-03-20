import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from helper import dataloader_output_to_tensor


class BaseDataset(Dataset):
    def __init__(self, configs, tok, triples, text_dict, mode='train', gt=None):
        super().__init__()
        self.configs = configs
        self.tok = tok
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.triples = triples
        self.mode = mode
        if gt is not None:
            self.train_tail_gt = gt['train_tail_gt']
            self.train_head_gt = gt['train_head_gt']

    def parse_ent_name(self, name):
        if self.configs.dataset == 'WN18RR':
            name = ' '.join(name.split(' , ')[:-2])
            return name
        return name or ''

    def construct_input_text(self, src_ent=None, rel=None, timestamp=None, predict='predict_tail'):
        src_name = self.ent_names[src_ent]
        rel_name = self.rel_names[rel]
        src_desc = ':' + self.ent_descs[src_ent] if self.configs.desc_max_length > 0 else ''

        timestamp = ' | ' + timestamp if timestamp else ''
        if predict == 'predict_tail':
            return src_name + ' ' + src_desc, rel_name + timestamp
        elif predict == 'predict_head':
            return src_name + ' ' + src_desc, 'reversed: ' + rel_name + timestamp
        else:
            raise ValueError('Mode is not correct!')

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = dataloader_output_to_tensor(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = dataloader_output_to_tensor(data, 'source_mask', padding_value=0)
        agg_data['ent_rel'] = dataloader_output_to_tensor(data, 'ent_rel')
        agg_data['tgt_ent'] = dataloader_output_to_tensor(data, 'tgt_ent', return_list=True)
        agg_data['triple'] = dataloader_output_to_tensor(data, 'triple', return_list=True)
        if self.mode == 'train':
            agg_data['labels'] = dataloader_output_to_tensor(data, 'labels').squeeze(-1)
        agg_data['pred_pos'] = dataloader_output_to_tensor(data, 'pred_pos', return_list=True)
        return agg_data


class CELossDataset(BaseDataset):
    def __init__(self, configs, tok, triples, text_dict, mode='train', gt=None):
        super().__init__(configs, tok, triples, text_dict, mode, gt)
        self.all_ent = set(range(configs.n_ent))

    def __len__(self):
        return len(self.triples) * 2 if self.mode == 'train' else len(self.triples)

    def __getitem__(self, index):
        if self.mode == 'train':
            mode = 'predict_tail' if index % 2 == 0 else 'predict_head'
            triple = self.triples[index // 2]
        else:
            mode = self.mode
            triple = self.triples[index]

        if not self.configs.is_temporal:
            head, tail, rel = triple
            timestamp = None
        else:
            head, tail, rel, timestamp = triple

        if mode == 'predict_tail':
            src = self.construct_input_text(src_ent=head,
                                            rel=rel,
                                            timestamp=timestamp,
                                            predict='predict_tail')
            tgt_ent = tail
        elif mode == 'predict_head':
            src = self.construct_input_text(src_ent=tail,
                                            rel=rel,
                                            timestamp=timestamp,
                                            predict='predict_head')
            tgt_ent = head
        else:
            raise ValueError('Mode is not correct!')

        ent_rel = (head, rel) if mode == 'predict_tail' else (tail, rel + self.configs.n_rel)
        src, text_pair = src
        tokenized_src = self.tok(src, text_pair=text_pair, max_length=self.configs.text_len, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        if 'bert' in self.configs.pretrained_model_name:
            mask_id = 103
        elif 'roberta' in self.configs.pretrained_model_name:
            mask_id = 50264
        source_ids.insert(-1, mask_id)
        source_mask.insert(-1, 1)

        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'triple': triple,
            'ent_rel': ent_rel,
            'tgt_ent': tgt_ent,
            'pred_pos': source_ids.index(mask_id),
        }

        if self.mode == 'train':
            out['labels'] = [tgt_ent]
        return out


class KGCDataModule(pl.LightningDataModule):
    def __init__(self, configs, train, valid, test, text_dict, tok, gt):
        super().__init__()
        self.configs = configs
        self.train = train
        self.valid = valid
        self.test = test
        # ent_names, rel_names .type: list
        self.text_dict = text_dict
        self.tok = tok
        self.gt = gt

        self.train_both = CELossDataset(configs, tok, train, text_dict, 'train', self.gt)
        self.valid_tail = CELossDataset(configs, tok, valid, text_dict, 'predict_tail')
        self.valid_head = CELossDataset(configs, tok, valid, text_dict, 'predict_head')
        self.test_tail = CELossDataset(configs, tok, test, text_dict, 'predict_tail')
        self.test_head = CELossDataset(configs, tok, test, text_dict, 'predict_head')

    def train_dataloader(self):
        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_tail_loader = DataLoader(self.valid_tail,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_tail.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        valid_head_loader = DataLoader(self.valid_head,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_head.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        return [valid_tail_loader, valid_head_loader]

    def test_dataloader(self):
        test_tail_loader = DataLoader(self.test_tail,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_tail.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        test_head_loader = DataLoader(self.test_head,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_head.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        return [test_tail_loader, test_head_loader]
