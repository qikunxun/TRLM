import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

def mask_matrix_dense(matrix, input_tripleid, batch_size, N, E):
    start = input_tripleid[:, 0:1]
    end = input_tripleid[:, 1:2]
    index = torch.arange(N).repeat(batch_size).reshape(batch_size, N).to(matrix.device)
    gte_start = start <= index
    lt_end = index < end
    mask = gte_start & lt_end
    mask[:, :E] = True
    matrix_new = matrix * mask.unsqueeze(dim=1)
    return matrix_new

device = torch.device('cuda')
class TRLMModel(nn.Module):
    def __init__(self, n, m, T, L, N, tau=0.2, use_gpu=False):
        super(TRLMModel, self).__init__()
        self.T = T
        self.L = L
        self.N = N
        self.n = n
        self.m = m
        self.tau = tau
        # self.w = nn.Parameter(torch.Tensor(self.n - 1, self.T, self.L, self.n))
        # nn.init.kaiming_uniform_(self.w, a=np.sqrt(5))
        #
        # self.weight = nn.Parameter(torch.Tensor(self.n - 1, self.L))
        # nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        self.w = nn.ParameterList()
        self.weight = nn.ParameterList()
        for i in range(self.n - 1):
            w = nn.Parameter(torch.Tensor(self.T, self.L, self.n))
            nn.init.kaiming_uniform_(w, a=np.sqrt(5))
            weight = nn.Parameter(torch.Tensor(self.L, 1))
            nn.init.zeros_(weight)
            self.w.append(w)
            self.weight.append(weight)

        self.use_gpu = use_gpu
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_x, input_r, input_triple2id, e2triple, triple2e, triple2r, triple2time,
                is_training=False, window=100):
        batch_size = input_x.shape[0]
        N = triple2e.shape[0]
        E = triple2e.shape[1]
        m = triple2time.shape[1]
        x_ori = torch.sparse.mm(input_x, e2triple)  # [b, N]
        time_transition = torch.triu(torch.ones(m, m, requires_grad=False), diagonal=0).to_sparse().to(x_ori.device)
        # time_transition_inv = torch.tril(torch.ones(m, m, requires_grad=False), diagonal=0).to_sparse().to(x_ori.device)

        states = []
        w_all = []
        weights = []
        for r in input_r:
            w_all.append(self.w[r])
            weights.append(self.weight[r])
        w_all = torch.stack(w_all, dim=0)
        weights = torch.stack(weights, dim=0)
        for t in range(self.T):
            w_probs = w_all[:, t, :, :]   # [b, L, n]
            # if flag: w_probs = self.w_inv[input_r][:, t, :, :]  # [b, L, n]
            w_probs = torch.softmax(w_probs, dim=-1)  # [b, L, n]
            if t == 0:
                x = x_ori.to_dense()  # [b, N]
                w = torch.permute(w_probs, (2, 0, 1))  # [n, b, L]
                s = torch.sparse.mm(triple2r, w.view(self.n, -1)).t().view(batch_size, self.L, -1)  # [b, L, N]
                s = torch.einsum('bm,blm->blm', x, s)  # [b, L, N]
                s = mask_matrix_dense(s, input_triple2id, batch_size, N, E)
                s = torch.sparse.mm(s.reshape(-1, N), triple2e).view(batch_size, self.L, E)  # [b, L, E]
                if is_training: s = self.dropout(s)
            if t >= 1:
                x = states[-1]  # [b, L, E]
                x_prev = torch.sparse.mm(x.reshape(-1, E), triple2e.transpose(1, 0))  # [b*L, N]
                x = torch.sparse.mm(x.reshape(-1, E), e2triple).view(batch_size, self.L, N)  # [b, L, N]
                mask = torch.sign(x_prev)  # [b*L, N]
                mask = torch.sparse.mm(mask, triple2time)  # [b*L, m]

                mask = torch.sparse.mm(mask, time_transition)
                # mask_ori = torch.sparse.mm(mask, time_transition)  # [b*L, m]
                # mask_inv = torch.sparse.mm(mask, time_transition_inv)  # [b*L, m]
                # flag = (input_r < (self.n - 1) // 2).bool().unsqueeze(dim=1)
                # flag = flag.repeat((1, self.L)).view(batch_size * self.L, 1)
                # mask = mask_ori * flag + mask_inv * ~flag
                mask = torch.sparse.mm(mask, triple2time.transpose(1, 0))  # [b*L, N]
                mask = self.activation(mask.reshape(batch_size, self.L, N))  # [b, L, N]
                mask[:, :, :E] = torch.ones_like(mask[:, :, :E])
                x = x * mask.detach()
                w = torch.permute(w_probs, (2, 0, 1))  # [n, b, L]
                s = torch.sparse.mm(triple2r, w.view(self.n, -1)).t().view(batch_size, self.L, -1)  # [b, L, N]
                s = torch.einsum('blm,blm->blm', x, s)  # [b, L, N]
                s = mask_matrix_dense(s, input_triple2id, batch_size, N, E)
                s = torch.sparse.mm(s.reshape(-1, N), triple2e).view(batch_size, self.L, E)  # [b, L, E]
                if is_training: s = self.dropout(s)
            s = self.activation(s)
            states.append(s)
        state = states[-1]

        weights = torch.sigmoid(weights.squeeze(dim=-1))
        s = torch.einsum('blm,bl->bm', state, weights)  # [b, E]
        return s



    def log_loss(self, p_score, label, logit_mask, thr=1e-20):
        one_hot = F.one_hot(torch.LongTensor([label]), p_score.shape[-1])
        if self.use_gpu:
            one_hot = one_hot.to(device)
            logit_mask = logit_mask.to(device)
        p_score = p_score - 1e30 * logit_mask.unsqueeze(dim=0)
        loss = -torch.sum(
            one_hot * torch.log(torch.maximum(F.softmax(p_score / self.tau, dim=-1), torch.ones_like(p_score) * thr)),
            dim=-1)
        return loss

    def activation(self, x):
        one = torch.autograd.Variable(torch.Tensor([1]))
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu:
            one = one.to(device)
            zero = zero.to(device)
        return torch.minimum(torch.maximum(x, zero), one)
