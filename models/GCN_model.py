import torch
from helper import *
from models.DisenLayer import DisenLayer
import pytorch_lightning as pl
from torch.nn.parallel import DataParallel

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        # print(x_samples.size())
        # print(y_samples.size())
        mu, logvar = self.get_mu_logvar(x_samples)

        return (-(mu - y_samples) ** 2 / 2. / logvar.exp()).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

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

class BaseModel(pl.LightningModule):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class CapsuleBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(CapsuleBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.init_embed = get_param((self.p.n_ent, self.p.embed_dim))
        if self.p.graph_model in ['transe', 'rotate']:
            self.init_rel = get_param((self.p.n_rel, self.p.embed_dim))
        elif self.p.graph_model in ['null', 'conve', 'distmult']:
            self.init_rel = get_param((self.p.n_rel * 2, self.p.embed_dim))
        self.pca = SparseInputLinear(self.p.embed_dim, self.p.num_factors * self.p.embed_dim)

        conv_ls = []
        for i in range(self.p.gcn_layer):
            conv = DisenLayer(self.edge_index, self.edge_type, self.p.embed_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p, head_num=self.p.head_num)
            # conv = DataParallel(conv)
            self.add_module('conv_{}'.format(i), conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls

        # if self.p.mi_train:
        if self.p.mi_method == 'club_b':
            num_dis = int((self.p.num_factors) * (self.p.num_factors - 1) / 2)
            # print(num_dis)
            self.mi_Discs = nn.ModuleList(
                [CLUBSample(self.p.embed_dim, self.p.embed_dim, self.p.embed_dim) for fac in range(num_dis)])
        elif self.p.mi_method == 'club_s':
            self.mi_Discs = nn.ModuleList(
                [CLUBSample((fac + 1) * self.p.embed_dim, self.p.embed_dim, (fac + 1) * self.p.embed_dim) for fac in
                 range(self.p.num_factors - 1)])

        self.register_parameter('bias', Parameter(torch.zeros(self.p.n_ent)))
        # self.rel_drop = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def mi_cal(self, sub_emb):

        def loss_dependence_club_s(sub_emb):
            mi_loss = 0.
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                mi_loss += self.mi_Discs[i](sub_emb[:, :bnd * self.p.embed_dim],
                                            sub_emb[:, bnd * self.p.embed_dim: (bnd + 1) * self.p.embed_dim])
            return mi_loss

        def loss_dependence_club_b(sub_emb):
            mi_loss = 0.
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    mi_loss += self.mi_Discs[cnt](sub_emb[:, i * self.p.embed_dim: (i + 1) * self.p.embed_dim],
                                                  sub_emb[:, j * self.p.embed_dim: (j + 1) * self.p.embed_dim])
                    cnt += 1
            return mi_loss

        if self.p.mi_method == 'club_s':
            mi_loss = loss_dependence_club_s(sub_emb)
        elif self.p.mi_method == 'club_b':
            mi_loss = loss_dependence_club_b(sub_emb)
        else:
            raise NotImplementedError

        return mi_loss

    def forward(self, sub, rel, mode='forward'):
        # if not self.p.no_enc:
        if self.p.graph_model in ['transe', 'rotate']:
            self.init_rel = torch.cat([self.rel_embed, -self.rel_embed], dim=0)
        x = self.act(self.pca(self.init_embed)).view(-1, self.p.num_factors, self.p.embed_dim)  # [N K F]
        r = self.init_rel
        for conv in self.conv_ls:
            x, r = conv(x, r)  # N K F
            x = self.drop(x)
        x = x.to(self.device)
        sub_emb = torch.index_select(x, 0, sub)
        if mode == 'forward':
            rel_emb = torch.index_select(self.init_rel, 0, rel).repeat(1, self.p.num_factors)
            rel_emb_single = torch.index_select(self.init_rel, 0, rel)
            sub_emb = sub_emb.view(-1, self.p.embed_dim * self.p.num_factors)
            mi_loss = self.mi_cal(sub_emb)
            return sub_emb, rel_emb, x, mi_loss, rel_emb_single
        else:
            return sub_emb   

    def training_step(self, batched_data, batch_idx):
        ent_rel = batched_data['ent_rel']
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        # tgt_ent -- .type: list
        tgt_ent = batched_data['tgt_ent']
        sub_emb = self(src_ent, rel, mode='train')
        lld_loss = 0.
        sub_emb = sub_emb.view(-1, self.p.num_factors * self.p.embed_dim)
        if self.p.mi_method == 'club_s':
            for i in range(self.p.num_factors - 1):
                bnd = i + 1
                lld_loss += self.mi_Discs[i].learning_loss(sub_emb[:, :bnd * self.p.embed_dim], sub_emb[:, bnd * self.p.embed_dim: (bnd + 1) * self.p.embed_dim])
        elif self.p.mi_method == 'club_b':
            cnt = 0
            for i in range(self.p.num_factors):
                for j in range(i + 1, self.p.num_factors):
                    lld_loss += self.mi_Discs[cnt].learning_loss(sub_emb[:, i * self.p.embed_dim: (i + 1) * self.p.embed_dim], sub_emb[:, j * self.p.embed_dim: (j + 1) * self.p.embed_dim])
        return lld_loss

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.parameters(), lr=self.p.lr)