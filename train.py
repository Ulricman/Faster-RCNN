import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.frcnn import FasterRCNN
from nets.frcnn_training import FasterRCNNTrainer, weights_init
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils_fit import fit_one_epoch
from utils.utils import get_lr
from torch.utils.tensorboard import SummaryWriter


Cuda = True
classes_path = '/SSD_DISK/users/yuanjunhao/FasterTorch/model_data/voc_classes.txt'
class_names = []
with open(classes_path) as f:
	for line in f:
		class_names.append(line.strip())
num_classes = len(class_names)
writer = SummaryWriter('./logs')

# ------------------------------------------------------------------------------------------------------------------
#   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
#   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
#
#   当model_path = ''的时候不加载整个模型的权值。
#
#   此处使用的是整个模型的权重，因此是在train.py进行加载的，下面的pretrain不影响此处的权值加载。
#   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，下面的pretrain = True，此时仅加载主干。
#   如果想要让模型从0开始训练，则设置model_path = ''，下面的pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
#   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
#
#   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
#   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
# ----------------------------------------------------------------------------------------------------------------------------#
model_path = ''
input_shape = [600, 600]
backbone = "resnet50"
# ----------------------------------------------------------------------------------------------------------------------------#
#   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
#   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
#   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
#   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
# ----------------------------------------------------------------------------------------------------------------------------#
pretrained = True
anchors_size = [8, 16, 32]

#   训练分为两个阶段，分别是冻结阶段和解冻阶段。
#   冻结阶段训练参数
#   此时模型的主干被冻结了，特征提取网络不发生改变
#   占用的显存较小，仅对网络进行微调
# ----------------------------------------------------#
Init_Epoch = 0
Freeze_Epoch = 30
Freeze_batch_size = 4
Freeze_lr = 1e-4
# ----------------------------------------------------#
#   解冻阶段训练参数
#   此时模型的主干不被冻结了，特征提取网络会发生改变
#   占用的显存较大，网络所有的参数都会发生改变
# ----------------------------------------------------#
UnFreeze_Epoch = 60
Unfreeze_batch_size = 2
Unfreeze_lr = 1e-5
Freeze_Train = False
num_workers = 4
#   获得图片路径和标签
train_annotation_path = '/SSD_DISK/users/yuanjunhao/FasterTorch/2007_train.txt'
val_annotation_path = '/SSD_DISK/users/yuanjunhao/FasterTorch/2007_val.txt'

#   获取classes和anchor
model = FasterRCNN(num_classes, anchor_scales=anchors_size, backbone=backbone, pretrained=pretrained)
if not pretrained:
	weights_init(model)
if model_path != '':
	print('Load weights {}.'.format(model_path))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_dict = model.state_dict()
	pretrained_dict = torch.load(model_path, map_location=device)  # load the mode saved by torch.save()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)

if Cuda:
	model_train = torch.nn.DataParallel(model)
	# automatically set up multiple GPUs for training.

	cudnn.benchmark = True
	model_train = model_train.cuda()
else:
	model_train = model.train()

with open(train_annotation_path) as f:
	train_lines = f.readlines()  # have the info of the bbox
with open(val_annotation_path) as f:
	val_lines = f.readlines()  # 2510
num_train = len(train_lines)  # 2501
num_val = len(val_lines)  # 2510

# ------------------------------------------------------#
#   也可以在训练初期防止权值被破坏。
#   Init_Epoch为起始世代
#   Freeze_Epoch为冻结训练的世代
#   Epoch总训练世代
#   提示OOM或者显存不足请调小Batch_size
# ------------------------------------------------------#
if True:
	batch_size = Freeze_batch_size
	lr = Freeze_lr
	start_epoch = Init_Epoch
	end_epoch = Freeze_Epoch

	epoch_step = num_train // batch_size
	epoch_step_val = num_val // batch_size

	if epoch_step == 0 or epoch_step_val == 0:
		raise ValueError("The dataset is too small for training, please expand the dataset.")

	optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

	train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
	val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
	gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
					 drop_last=True, collate_fn=frcnn_dataset_collate)  # 625
	# ‘pin_memory’:If true, the data loader will copy tensors into CUDA pinned memory before returning them.
	# 'collate_fn': merges a list of samples to form a mini-batch.
	gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
						 drop_last=True, collate_fn=frcnn_dataset_collate)

	# ------------------------------------#
	#   冻结一定部分训练
	# ------------------------------------#
	if Freeze_Train:
		for param in model.extractor.parameters():
			param.requires_grad = False

	model.freezeBN()

	train_util = FasterRCNNTrainer(model, optimizer)

	for epoch in range(start_epoch, end_epoch):
		total_loss, val_loss = 0, 0
		print('Start Train')
		with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
			for iteration, batch in enumerate(gen):
				if iteration >= epoch_step:
					break
				images, boxes, labels = batch[0], batch[1], batch[2]
				with torch.no_grad():
					images = torch.from_numpy(images).type(torch.FloatTensor)
					if Cuda:
						images = images.cuda()

				rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1)
				# total is the addition of (rpn_loc, rpn_cls, roi_cls, roi_cls).

				# .item() can convert the one element tensors to Python scalars.
				total_loss += total.item()

				pbar.set_postfix(**{'train_loss': '{:.2f}'.format(total_loss / (iteration + 1))})
				pbar.update(1)

		print('Finish Train')

		print('Start Validation')
		with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
			for iteration, batch in enumerate(gen_val):
				if iteration >= epoch_step_val:
					break
				images, boxes, labels = batch[0], batch[1], batch[2]
				with torch.no_grad():
					images = torch.from_numpy(images).type(torch.FloatTensor)
					if Cuda:
						images = images.cuda()

					train_util.optimizer.zero_grad()
					_, _, _, _, val_total = train_util.forward(images, boxes, labels, 1)

					val_loss += val_total.item()
					pbar.set_postfix(**{'val_loss': '{:.2f}'.format(val_loss / (iteration + 1))})
					pbar.update(1)

		print('Finish Validation')
		writer.add_scalar('total_loss/loss', total_loss / epoch_step, epoch + 1)
		writer.add_scalar('val_loss/loss', val_loss / epoch_step_val, epoch + 1)
		print('Epoch:' + str(epoch + 1) + '/' + str(end_epoch))
		print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
		torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (
			epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))

		lr_scheduler.step()

if True:
	batch_size = Unfreeze_batch_size
	lr = Unfreeze_lr
	start_epoch = Freeze_Epoch
	end_epoch = UnFreeze_Epoch

	epoch_step = num_train // batch_size
	epoch_step_val = num_val // batch_size

	if epoch_step == 0 or epoch_step_val == 0:
		raise ValueError("The dataset is too small for training, please expand the dataset.")

	optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

	train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
	val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
	gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
					 drop_last=True, collate_fn=frcnn_dataset_collate)
	gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
						 drop_last=True, collate_fn=frcnn_dataset_collate)

	if Freeze_Train:
		for param in model.extractor.parameters():
			param.requires_grad = True

	model.freezeBN()

	train_util = FasterRCNNTrainer(model, optimizer)

	for epoch in range(start_epoch, end_epoch):

		total_loss, val_loss = 0, 0
		print('Start Train')
		with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
			for iteration, batch in enumerate(gen):
				if iteration >= epoch_step:
					break
				images, boxes, labels = batch[0], batch[1], batch[2]
				with torch.no_grad():
					images = torch.from_numpy(images).type(torch.FloatTensor)
					if Cuda:
						images = images.cuda()

				total = train_util.train_step(images, boxes, labels, 1)[0]
				total_loss += total.item()

				pbar.set_postfix(**{'train_loss': '{:.2f}'.format(total_loss / (iteration + 1))})
				pbar.update(1)

		print('Finish Train')

		print('Start Validation')
		with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
			for iteration, batch in enumerate(gen_val):
				if iteration >= epoch_step_val:
					break
				images, boxes, labels = batch[0], batch[1], batch[2]
				with torch.no_grad():
					images = torch.from_numpy(images).type(torch.FloatTensor)
					if Cuda:
						images = images.cuda()

					train_util.optimizer.zero_grad()
					val_total = train_util.forward(images, boxes, labels, 1)[0]

					val_loss += val_total.item()
					pbar.set_postfix(**{'val_loss': '{:.2f}'.format(val_loss / (iteration + 1))})
					pbar.update(1)

		print('Finish Validation')
		writer.add_scalar('total_loss/loss', total_loss / epoch_step, epoch + 1)
		writer.add_scalar('val_loss/loss', val_loss / epoch_step_val, epoch + 1)
		print('Epoch:' + str(epoch + 1) + '/' + str(end_epoch))
		print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
		torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (
			epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))

		lr_scheduler.step()

writer.close()
