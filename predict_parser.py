# New
import time
import cv2
import numpy as np
from PIL import Image
from frcnn import FRCNN
import os
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--crop', dest='crop', default=False)
parser.add_option('--path', dest='path', help='The path to the current file.')
parser.add_option('--img', dest='img', help='The path to the current file.')
parser.add_option('--save', dest='save', help='The path to save the predicted image.')
parser.add_option('--backbone', dest='backbone', help='The backbone to be used when predicting.', default='resnet50')
parser.add_option('--model', dest='model', help='The trained mode.')

(options, args) = parser.parse_args()

if not options.root_path:
	parser.error('Error:path to the current file must be specified. Pass --path to command line')

root_path = options.path

if not options.img:
	parser.error('Error:path to the image to predict must be specified. Pass --img to the command line')

img = options.img

if not options.save:
	print('The predicted image will be saved at the same location as the original image.')
	save_path = os.path.dirname(img)
else:
	save_path = options.save
# crop = False

if not options.model:
	parser.error('Error:the trained model when predicting must be specified. Path --model to the command line')

model = options.model

crop = bool(options.crop)

# weights_name = 'weights/voc_weights_resnet.pth'

# root_path = '/SSD_DISK/users/yuanjunhao/FasterTorch/New'

# classes_path = os.path.join(root_path, 'model_data/voc_classes.txt')
# model_path = os.path.join(root_path, weights_name)
# frcnn = FRCNN(model_path=model_path, classes_path=classes_path)
frcnn = FRCNN(root=root_path)

# img = 'img/3.jpg'


image = Image.open(img)
r_image = frcnn.detect_image(image, crop=crop)
r_image.save('img/3_out.jpg')
print('The predicted image has been saved.')
