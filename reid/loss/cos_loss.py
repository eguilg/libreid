from __future__ import absolute_import

import torch
import torch.nn as nn


class CosLoss(nn.Module):
	def __init__(self):
		super(CosLoss, self).__init__()
		self.cross_entropy = nn.CrossEntropyLoss()
		self.m = 0.35
		self.s = 128

	def forward(self, inputs, targets):

		# batch_size = inputs.size(0);
		# loss = None
		# for i in range(batch_size):
		# 	l = - torch.log(torch.exp(self.s * (inputs[i][targets[i]] - self.m)) /
		# 					 torch.exp(self.s * (inputs[i] - self.m).sum()))
		#
		#
		# 	if loss is None:
		# 		loss = l
		# 	else:
		# 		loss += l
		#
		# loss = loss.sum() / batch_size
		loss = self.cross_entropy(self.s*inputs, targets)

		return loss
