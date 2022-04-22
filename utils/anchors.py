import numpy as np


# anchor base
# [[ -45.254833  -90.50967    45.254833   90.50967 ]
#  [ -90.50967  -181.01933    90.50967   181.01933 ]
#  [-181.01933  -362.03867   181.01933   362.03867 ]
#  [ -64.        -64.         64.         64.      ]
#  [-128.       -128.        128.        128.      ]
#  [-256.       -256.        256.        256.      ]
#  [ -90.50967   -45.254833   90.50967    45.254833]
#  [-181.01933   -90.50967   181.01933    90.50967 ]
#  [-362.03867  -181.01933   362.03867   181.01933 ]]


def generate_anchor_base(base_size=16, ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32)):  # 128, 256, 512
	anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
	for i in range(len(ratios)):
		for j in range(len(anchor_scales)):
			h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
			w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

			index = i * len(anchor_scales) + j
			anchor_base[index, 0] = - h / 2.
			anchor_base[index, 1] = - w / 2.
			anchor_base[index, 2] = h / 2.
			anchor_base[index, 3] = w / 2.
	return anchor_base


# --------------------------------------------#
#   对基础先验框进行拓展对应到所有特征点上
# --------------------------------------------#

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
	# ---------------------------------#
	#   计算网格中心点
	# ---------------------------------#
	shift_x = np.arange(0, width * feat_stride, feat_stride)  # (38,)
	shift_y = np.arange(0, height * feat_stride, feat_stride)
	shift_x, shift_y = np.meshgrid(shift_x, shift_y)
	# (38, 38), shift_x[0] = [  0  16  32 ... 560 576 592].
	shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)  # (38 * 38, 4)

	# ---------------------------------#
	#   每个网格点上的9个先验框
	# ---------------------------------#
	A = anchor_base.shape[0]  # 9
	K = shift.shape[0]  # 1444
	anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))  # (38 * 38, 9, 4)
	# ---------------------------------#
	#   所有的先验框
	# ---------------------------------#
	anchor = anchor.reshape((K * A, 4)).astype(np.float32)  # (38 * 38 * 9, 4)
	# the coordinates are of midpoints on the edges.
	return anchor
