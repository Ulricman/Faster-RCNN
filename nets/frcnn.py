import torch.nn as nn
from nets.classifier import Resnet50RoIHead
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork


class FasterRCNN(nn.Module):
	def __init__(self, num_classes, mode="training", feat_stride=16, anchor_scales=(8, 16, 32), ratios=(0.5, 1, 2),
				 backbone='resnet50', pretrained=True):
		super(FasterRCNN, self).__init__()
		self.feat_stride = feat_stride

		if backbone == 'resnet50':
			self.extractor, classifier = resnet50(pretrained=pretrained)
			self.rpn = RegionProposalNetwork(in_channels=1024, mid_channels=512, ratios=ratios,
											 anchor_scales=anchor_scales, feat_stride=self.feat_stride, mode=mode)
			self.head = Resnet50RoIHead(n_class=num_classes + 1, roi_size=14, spatial_scale=1, classifier=classifier)

	def forward(self, x, scale=1.):
		img_size = x.shape[2:]

		# use the backbone (resnet) to extract features.
		base_feature = self.extractor.forward(x)  # (4, 1024, 38, 38)

		# get the fine tuned rois.
		_, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
		# ---------------------------------------#
		#   获得classifier的分类结果和回归结果
		# ---------------------------------------#
		roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
		return roi_cls_locs, roi_scores, rois, roi_indices

	def freezeBN(self):
		for module in self.modules():
			if isinstance(module, nn.BatchNorm2d):
				module.eval()
