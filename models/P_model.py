import os
import time
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoConfig
from helper import get_param, get_performance, get_loss_fn, GRAPH_MODEL_CLASS
from models.prompter import Prompter
from models.bert_for_layerwise import BertModelForLayerwise
from models.roberta_for_layerwise import RobertaModelForLayerwise
from models.AutomaticWeightedLoss import AutomaticWeightedLoss
from models.GCN_model import CapsuleBase


class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias

class KGCPromptTuner(pl.LightningModule):
    def __init__(self, configs, text_dict, gt, GCN_model):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.all_tail_gt = gt['all_tail_gt']
        self.all_head_gt = gt['all_head_gt']

        self.plm_configs = AutoConfig.from_pretrained(configs.pretrained_model)
        self.plm_configs.prompt_length = self.configs.prompt_length
        self.plm_configs.prompt_hidden_dim = self.configs.prompt_hidden_dim
        if 'roberta' in configs.pretrained_model_name:
            self.plm = RobertaModelForLayerwise.from_pretrained(configs.pretrained_model)
        elif 'bert' in configs.pretrained_model_name:
            self.plm = BertModelForLayerwise.from_pretrained(configs.pretrained_model)

        self.prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)
        self.fc = nn.Linear(configs.prompt_length * self.plm_configs.hidden_size, configs.embed_dim)
        if configs.prompt_length > 0:
            for p in self.plm.parameters():
                p.requires_grad = False

        self.graph_model = GRAPH_MODEL_CLASS[self.configs.graph_model](configs)

        self.history = {'perf': ..., 'loss': [], 'lld_loss': []}
        self.loss_fn = get_loss_fn(configs)
        self._MASKING_VALUE = -1e4 if self.configs.use_fp16 else -1e9
        if self.configs.alpha_step > 0:
            self.alpha_corr = 0.
        else:
            self.alpha_corr = self.configs.alpha_corr

        # text prediction
        ent_text_embeds_file = '{}/{}/entity_embeds_{}.pt'.format(self.configs.dataset_path, self.configs.dataset, self.configs.pretrained_model_name.lower())
        self.ent_text_embeds = torch.load(ent_text_embeds_file)
        self.ent_transform = torch.nn.Linear(self.plm_configs.hidden_size, self.plm_configs.hidden_size)

        # weighted_loss
        self.loss_weight = AutomaticWeightedLoss(2)

        # GCN pre-process
        self.GCN = GCN_model
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, ent_rel, src_ids, src_mask, pred_pos):
        bs = ent_rel.size(0)

        ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        # ent_embed = all_ent_embed[ent]
        # rel_embed = all_rel_embed[rel]
        ent_embed, rel_embed, all_ent_embed, corr, rel_embed_single = self.GCN(ent, rel)

        all_ent_embed = all_ent_embed.view(-1, self.configs.num_factors, self.configs.embed_dim)
        ent_embed = ent_embed.view(-1, self.configs.num_factors, self.configs.embed_dim)
        rel_embed_exd = torch.unsqueeze(rel_embed_single, 1)
        embed_input = torch.cat([ent_embed, rel_embed_exd], dim=1)

        prompt = self.prompter(embed_input)
        prompt_attention_mask = torch.ones(ent_embed.size(0), self.configs.prompt_length * (self.configs.num_factors + 1)).type_as(src_mask)
        src_mask = torch.cat((prompt_attention_mask, src_mask), dim=1)
        output = self.plm(input_ids=src_ids, attention_mask=src_mask, layerwise_prompt=prompt)

        # last_hidden_state -- .shape: (batch_size, seq_len, model_dim)
        last_hidden_state = output.last_hidden_state

        ent_rel_state = last_hidden_state[:, :self.configs.prompt_length * (self.configs.num_factors + 1)]
        plm_embeds = torch.chunk(ent_rel_state, chunks=(self.configs.num_factors + 1), dim=1)
        plm_ent_embeds, plm_rel_embed = plm_embeds[:self.configs.num_factors], plm_embeds[-1]

        plm_ent_embed = torch.stack(plm_ent_embeds, dim=1)
        plm_ent_embed = self.fc(plm_ent_embed.reshape(ent_embed.size(0), self.configs.num_factors, -1))
        plm_rel_embed = self.fc(plm_rel_embed.reshape(rel_embed.size(0), -1)).repeat(1, self.configs.num_factors)
        plm_rel_embed = plm_rel_embed.view(-1, self.configs.num_factors, self.configs.embed_dim)
         
        attention = self.leakyrelu(torch.einsum('bkf,bkf->bk', [plm_ent_embed, plm_rel_embed]))
        attention = nn.Softmax(dim=-1)(attention)

        # pred -- .shape: (batch_size, num_factors, embed_dim)         
        pred = self.graph_model(plm_ent_embed, plm_rel_embed).view(-1, self.configs.num_factors, self.configs.embed_dim)
        # logits -- .shape: (batch_size, num_factors, n_ent)
        logits = self.graph_model.get_logits(pred, all_ent_embed)
        logits = torch.einsum('bk,bkn->bn', [attention, logits])

        # text prediction
        mask_token_state = []
        for i in range(ent.size(0)):
            pred_embed = last_hidden_state[i, pred_pos[i]]
            mask_token_state.append(pred_embed)

        mask_token_state = torch.stack(mask_token_state, dim=0)
         
        output_tmp = self.ent_transform(mask_token_state)
        self.ent_text_embeds = self.ent_text_embeds.to(output_tmp.device)
        output = torch.einsum('bf,nf->bn', [output_tmp, self.ent_text_embeds])

        return logits, output, corr

    def training_step(self, batched_data, batch_idx, optimizer_idx):
        if self.configs.alpha_step > 0 and self.alpha_corr < self.configs.alpha_corr:
            self.alpha_corr = min(self.alpha_corr + self.configs.alpha_step, self.configs.alpha_corr)
        if optimizer_idx == 0:
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
            src_ids = batched_data['source_ids']
            src_mask = batched_data['source_mask']
            # ent_rel .shape: (batch_size, 2)
            ent_rel = batched_data['ent_rel']
            tgt_ent = batched_data['tgt_ent']
            labels = batched_data['labels']
            pred_pos = batched_data['pred_pos']
            
            logits, output, corr = self(ent_rel, src_ids, src_mask, pred_pos)
            struc_loss = self.loss_fn(logits, labels)
            lm_loss = self.loss_fn(output, labels)
            loss = struc_loss + lm_loss + self.alpha_corr * corr

            self.history['loss'].append(loss.detach().item())
            return loss
        elif optimizer_idx == 1:
            lld_loss = self.GCN.training_step(batched_data, batch_idx)
            self.history['lld_loss'].append(lld_loss.detach().item())
            return lld_loss


    def validation_step(self, batched_data, batch_idx, dataset_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        # test_triples .shape: (batch_size, 3)
        test_triples = batched_data['triple']
        # ent_rel .shape: (batch_size, 2)
        ent_rel = batched_data['ent_rel']
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        # tgt_ent -- .type: list
        tgt_ent = batched_data['tgt_ent']
        pred_pos = batched_data['pred_pos']
        gt = self.all_tail_gt if dataset_idx == 0 else self.all_head_gt
        logits, outputs, _ = self(ent_rel, src_ids, src_mask, pred_pos)
        logits = logits.detach()
        outputs = outputs.detach()
        combines = self.loss_weight(logits, outputs)
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            if self.configs.is_temporal:
                tgt_filter = gt[(hi, ri, test_triples[i][3])]
            else:
                # tgt_filter .type: list()
                tgt_filter = gt[(hi, ri)]
            ## store target score
            tgt_struc_score = logits[i, ti].item()
            tgt_lm_score = outputs[i, ti].item()
            tgt_combine_score = combines[i, ti].item()
            ## remove the scores of the entities we don't care
            logits[i, tgt_filter] = self._MASKING_VALUE
            outputs[i, tgt_filter] = self._MASKING_VALUE
            combines[i, tgt_filter] = self._MASKING_VALUE
            ## recover the target values
            logits[i, ti] = tgt_struc_score
            outputs[i, ti] = tgt_lm_score
            combines[i, ti] = tgt_combine_score
        _, struc_argsort = torch.sort(logits, dim=1, descending=True)
        _, lm_argsort = torch.sort(outputs, dim=1, descending=True)
        _, combine_argsort = torch.sort(combines, dim=1, descending=True)
        struc_argsort = struc_argsort.cpu().numpy()
        lm_argsort = lm_argsort.cpu().numpy()
        combine_argsort = combine_argsort.cpu().numpy()

        struc_ranks = []
        lm_ranks = []
        combine_ranks = []
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            struc_rank = np.where(struc_argsort[i] == ti)[0][0] + 1
            lm_rank = np.where(lm_argsort[i] == ti)[0][0] + 1
            combine_rank = np.where(combine_argsort[i] == ti)[0][0] + 1
            struc_ranks.append(struc_rank)
            lm_ranks.append(lm_rank)
            combine_ranks.append(combine_rank)
        if self.configs.use_log_ranks:
            filename = os.path.join(self.configs.save_dir, f'Epoch-{self.current_epoch}-ranks.tmp')
            self.log_ranks(filename, test_triples, struc_argsort, struc_ranks, batch_idx, 'struc')
            self.log_ranks(filename, test_triples, lm_argsort, lm_ranks, batch_idx, 'lm')
            self.log_ranks(filename, test_triples, combine_argsort, combine_ranks, batch_idx, 'combine')

        return struc_ranks, lm_ranks, combine_ranks

    def validation_epoch_end(self, outs):
        tail_ranks = outs[0]
        head_ranks = outs[1]

        struc_tail_ranks = []
        lm_tail_ranks = []
        combine_tail_ranks = []
        struc_head_ranks = []
        lm_head_ranks = []
        combine_head_ranks = []

        for struc_rank, lm_rank, combine_rank in tail_ranks:
            if self.configs.distributed_training:
                struc_rank_tensor = []
                lm_rank_tensor = []
                combine_rank_tensor = []
                for ranks in struc_rank:
                    for rank in ranks:
                        struc_rank_tensor.append(int(rank))
                for ranks in lm_rank:
                    for rank in ranks:
                        lm_rank_tensor.append(int(rank))
                for ranks in combine_rank:
                    for rank in ranks:
                        combine_rank_tensor.append(int(rank))
                struc_rank = struc_rank_tensor
                lm_rank = lm_rank_tensor
                combine_rank = combine_rank_tensor

            struc_tail_ranks.append(struc_rank)
            lm_tail_ranks.append(lm_rank)
            combine_tail_ranks.append(combine_rank)
        for struc_rank, lm_rank, combine_rank in head_ranks:
            if self.configs.distributed_training:
                struc_rank_tensor = []
                lm_rank_tensor = []
                combine_rank_tensor = []
                for ranks in struc_rank:
                    for rank in ranks:
                        struc_rank_tensor.append(int(rank))
                for ranks in lm_rank:
                    for rank in ranks:
                        lm_rank_tensor.append(int(rank))
                for ranks in combine_rank:
                    for rank in ranks:
                        combine_rank_tensor.append(int(rank))
                struc_rank = struc_rank_tensor
                lm_rank = lm_rank_tensor
                combine_rank = combine_rank_tensor

            struc_head_ranks.append(struc_rank)
            lm_head_ranks.append(lm_rank)
            combine_head_ranks.append(combine_rank)
            
        struc_tail_ranks = np.concatenate(struc_tail_ranks)
        struc_head_ranks = np.concatenate(struc_head_ranks)
        lm_tail_ranks = np.concatenate(lm_tail_ranks)
        lm_head_ranks = np.concatenate(lm_head_ranks)
        combine_tail_ranks = np.concatenate(combine_tail_ranks)
        combine_head_ranks = np.concatenate(combine_head_ranks)

        struc_perf = get_performance(self, struc_tail_ranks, struc_head_ranks, 'struc')
        lm_perf = get_performance(self, lm_tail_ranks, lm_head_ranks, 'text')
        combine_perf = get_performance(self, combine_tail_ranks, combine_head_ranks, 'combine')
        print('Epoch: ', self.current_epoch)
        print('struc:\n', struc_perf)
        print('text:\n', lm_perf)
        print('combine:\n', combine_perf)

    def test_step(self, batched_data, batch_idx, dataset_idx):
        return self.validation_step(batched_data, batch_idx, dataset_idx)

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        mi_disc_params = list(map(id, self.GCN.mi_Discs.parameters()))
        rest_params = filter(lambda x: id(x) not in mi_disc_params, self.parameters())
        return [torch.optim.AdamW(rest_params, lr=self.configs.lr), torch.optim.AdamW(self.GCN.mi_Discs.parameters(), lr=self.configs.lr)]

    def log_ranks(self, filename, test_triples, argsort, ranks, batch_idx, mode):
        assert len(test_triples) == len(ranks), 'length mismatch: test_triple, ranks!'
        with open(filename, 'a') as file:
            for i, triple in enumerate(test_triples):
                if not self.configs.is_temporal:
                    head, tail, rel = triple
                    timestamp = ''
                else:
                    head, tail, rel, timestamp = triple
                    timestamp = ' | ' + timestamp
                rank = ranks[i].item()
                triple_str = self.ent_names[head] + ' [' + self.ent_descs[head] + '] | ' + self.rel_names[rel]\
                    + ' | ' + self.ent_names[tail] + ' [' + self.ent_descs[tail] + '] ' + timestamp + '(%d %d %d)' % (head, tail, rel)
                file.write(str(batch_idx * self.configs.val_batch_size + i) + '. ' + triple_str + '=> ' + mode + '_ranks: ' + str(rank) + '\n')

                best10 = argsort[i, :10]
                for ii, ent in enumerate(best10):
                    ent = ent.item()
                    mark = '*' if (ii + 1) == rank else ' '
                    file.write('\t%2d%s ' % (ii + 1, mark) + self.ent_names[ent] + ' [' + self.ent_descs[ent] + ']' + ' (%d)' % ent + '\n')
