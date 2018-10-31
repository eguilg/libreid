from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class CoupleClusterLoss(nn.Module):
    def __init__(self, margin=0):
        super(CoupleClusterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_cp, dist_cn = [], []
        for i in range(n):
            x_p = inputs[mask[i]]
            x_n = inputs[mask[i] == 0]

            center = torch.mean(x_p, dim=0)
            dist_cp.append(torch.pow(x_p - center, 2).sum(dim=1, keepdim=True).max())
            dist_cn.append(torch.pow(x_n - center, 2).sum(dim=1, keepdim=True).min())
        dist_cp = torch.stack(dist_cp)
        dist_cn = torch.stack(dist_cn)

        # Compute ranking hinge loss
        y = dist_cn.data.new()
        y.resize_as_(dist_cn.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_cn, dist_cp, y)
        prec = (dist_cn.data > dist_cp.data).sum() * 1. / y.size(0)
        return loss, prec
