import warnings
import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")


def normal_init(m, mean, stddev, truncated=False):
	if truncated:
		m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
	else:
		m.weight.data.normal_(mean, stddev)
		m.bias.data.zero_()


class Resnet50RoIHead(nn.Module):
	def __init__(self, n_class, roi_size, spatial_scale, classifier):
		super(Resnet50RoIHead, self).__init__()
		self.classifier = classifier
		# --------------------------------------#
		#   对ROIPooling后的的结果进行回归预测
		# --------------------------------------#
		self.cls_loc = nn.Linear(2048, n_class * 4)
		# -----------------------------------#
		#   对ROIPooling后的的结果进行分类
		# -----------------------------------#
		self.score = nn.Linear(2048, n_class)
		# -----------------------------------#
		#   权值初始化
		# -----------------------------------#
		normal_init(self.cls_loc, 0, 0.001)
		normal_init(self.score, 0, 0.01)

		self.roi = RoIPool((roi_size, roi_size), spatial_scale)

	def forward(self, base_feature, rois, roi_indices, img_size):
		"""
		:param base_feature: torch.Size([1, 1024, 38, 38]), because the input is just one image.
		:param rois: torch.Size([128, 4]), is the sample_roi with the size of 128.
		:param roi_indices: torch.Size([128])
		:param img_size: torch.Size([600, 600])
		:return roi_cls_locs: torch.Size([1, 128, 84])
		:return roi_scores: torch.Size([1, 128, 21])
		"""
		n = base_feature.shape[0]  # batch_size
		if base_feature.is_cuda:
			roi_indices = roi_indices.cuda()
			rois = rois.cuda()

		# find the related feature of the roi on the base_feature.
		rois_feature_map = torch.zeros_like(rois)
		rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * base_feature.size()[3]
		rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * base_feature.size()[2]

		indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)  # (128, 5)
		# the first column is the indices, and the other four columns are rois.

		# intercept the base_feature by the rois.

		# the "indices_and_rois" contains the coordinates of the related feature on the base_feature.
		ROIPooling = self.roi(base_feature, indices_and_rois)  # (128, 1024, 14, 14)
		# -----------------------------------#
		#   利用classifier网络进行特征提取
		# -----------------------------------#
		fc7 = self.classifier(ROIPooling)  # (128, 2048, 1, 1)
		# there is a global average pooling in the classifier, and the shape before the pooling is (128, 2048, 7, 7).

		fc7 = fc7.view(fc7.size(0), -1)  # (128, 2048)

		roi_cls_locs = self.cls_loc(fc7)  # (128, 84)
		roi_scores = self.score(fc7)  # (128, 21)
		roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))  # (1, 128, 84)
		roi_scores = roi_scores.view(n, -1, roi_scores.size(1))  # (1, 128, 21)
		return roi_cls_locs, roi_scores
