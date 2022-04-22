import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


def normal_init(m, mean, stddev, truncated=False):
	if truncated:
		m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
	else:
		m.weight.data.normal_(mean, stddev)
		m.bias.data.zero_()


class ProposalCreator:
	def __init__(self, mode, nms_iou=0.7, n_train_pre_nms=12000, n_train_post_nms=600, n_test_pre_nms=3000,
				 n_test_post_nms=300, min_size=16):
		self.mode = mode  # set to train/prediction
		self.nms_iou = nms_iou

		# the number of proposals when training.
		self.n_train_pre_nms = n_train_pre_nms
		self.n_train_post_nms = n_train_post_nms
		# the number of proposals when predicting.
		self.n_test_pre_nms = n_test_pre_nms
		self.n_test_post_nms = n_test_post_nms
		self.min_size = min_size

	def __call__(self, loc, score, anchor, img_size, scale=1.):
		"""
		:param loc: the fine tuning value of the original anchors. (dx, dy, dw, dh), (12996, 4)
		:param score: the score (probability) of whether the anchor (?) contains an object or not. (12996,)
		:param anchor: the coordinates of the base anchors. (12996, 4)
		:param img_size: (600, 600)
		:param scale: 1
		:return: the selected rois after NMS.
		"""
		if self.mode == "training":
			n_pre_nms = self.n_train_pre_nms
			n_post_nms = self.n_train_post_nms
		else:
			n_pre_nms = self.n_test_pre_nms
			n_post_nms = self.n_test_post_nms

		anchor = torch.from_numpy(anchor)  # turn anchor into tensors.
		if loc.is_cuda:
			anchor = anchor.cuda()
		# turn the predicted results by RPN into roi.
		# "loc" is the fine tuning value of the (x, y, w, h).
		roi = loc2bbox(anchor, loc)  # (12996, 4)

		#  prevent the roi from going beyond the boundary.
		roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
		roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

		# the height and width should not less than 16.
		min_size = self.min_size * scale
		keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
		roi = roi[keep, :]
		score = score[keep]

		# select rois with the n_pre_nms highest score.
		order = torch.argsort(score, descending=True)
		order = order[:n_pre_nms]
		roi = roi[order, :]
		score = score[order]

		keep = nms(roi, score, self.nms_iou)
		keep = keep[:n_post_nms]
		# select 300/600 best rois.
		roi = roi[keep]
		return roi


class RegionProposalNetwork(nn.Module):
	def __init__(self, in_channels=1024, mid_channels=512, ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32), feat_stride=16,
				 mode="training"):
		super(RegionProposalNetwork, self).__init__()
		# generate base anchors for one point (shape is (9, 4))
		self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
		n_anchor = self.anchor_base.shape[0]  # 9

		self.conv1 = nn.Conv2d(in_channels, mid_channels, (3, 3), (1, 1), 1)
		# only change the channels from 1024 to 512.

		self.score = nn.Conv2d(mid_channels, n_anchor * 2, (1, 1), (1, 1), 0)
		self.loc = nn.Conv2d(mid_channels, n_anchor * 4, (1, 1), (1, 1), 0)

		# the distance between feature points measured in the original image.
		self.feat_stride = feat_stride
		# -----------------------------------------#
		#   用于对建议框解码并进行非极大抑制
		# -----------------------------------------#
		self.proposal_layer = ProposalCreator(mode=mode)

		# weight initialization for RPN.
		normal_init(self.conv1, 0, 0.01)
		normal_init(self.score, 0, 0.01)
		normal_init(self.loc, 0, 0.01)

	def forward(self, x, img_size, scale=1.):
		"""
		:param x: the shared feature map.
		:param img_size:
		:param scale: 1
		:return rpn_locs: the fine tuning value of the original anchors. (4, 12996, 4)
		:return rpn_scores: the score (probability) of the roi to contain an object. (4, 12996, 2)
		:return rois: all the roi in the minibatch (n). (600 * 4, 4)
		:return roi_indices: a tensor to indicate which image the roi belongs to. (600 * 4,)
		:return anchor: all of the base anchors. (38 * 38 * 9, 4)
		"""
		n, _, h, w = x.shape  # (4, 1024, 38, 38)
		x = F.relu(self.conv1(x))  # (4, 512, 38, 38)

		rpn_locs = self.loc(x)  # (4, 9 * 4, 38, 38)
		rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # (4, 12996, 4)

		rpn_scores = self.score(x)  # (4, 9 * 2, 38, 38)
		rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)  # (4, 12996, 2)
		# 38 * 38 * 9 = 12996 is the number of all the anchors in one feature map.

		# --------------------------------------------------------------------------------------#
		#   进行softmax概率计算，每个先验框只有两个判别结果
		#   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
		# --------------------------------------------------------------------------------------#
		rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)  # (4, 12996, 2)
		rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()  # (4, 12996)
		rpn_fg_scores = rpn_fg_scores.view(n, -1)  # (4, 12996)

		# ------------------------------------------------------------------------------------------------#
		#   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
		# ------------------------------------------------------------------------------------------------#

		# generate all the anchors in the feature map.
		anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)  # (12996, 4)
		# the coordinates in the anchor are of points on the edges.

		rois = list()
		roi_indices = list()
		for i in range(n):  # n is the batch size.
			# select some roi by NMS. (tensor)
			roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)  # (600, 4)
			# batch_index = i * torch.ones((len(roi),))
			batch_index = torch.Tensor([i]) * torch.ones((len(roi),))
			rois.append(roi)
			roi_indices.append(batch_index)

		rois = torch.cat(rois, dim=0)  # (600 * 4, 4)
		roi_indices = torch.cat(roi_indices, dim=0)  # (600 * 4,)

		return rpn_locs, rpn_scores, rois, roi_indices, anchor
