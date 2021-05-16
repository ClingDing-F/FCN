import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from evalution_segmentaion import eval_semantic_segmentation
from datasets import ImageDatasets
from FCN import FCN
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 4
miou_list = [0]

TEST_ROOT = './data/train'
TEST_LABEL = './data/train_labels'
CLASS_DICT_PATH = './data/class_dict.csv'
crop_size = (224, 352)

trans_image = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

Image_test = ImageDatasets([TEST_ROOT, TEST_LABEL, CLASS_DICT_PATH], crop_size, trans_image)
DataLoader_test = DataLoader(Image_test, batch_size=1, shuffle=True, num_workers=0)

net = FCN(12).to(device)
net.load_state_dict(torch.load("xxx.path"))
net.eval()

pd_label_color = pd.read_csv('./data/class_dict.csv', sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
	tmp = pd_label_color.iloc[i]
	color = [tmp['r'], tmp['g'], tmp['b']]
	colormap.append(color)

cm = np.array(colormap).astype('uint8')

dir = "./result/"

for i, sample in enumerate(DataLoader_test):
	valImg = sample['img'].to(device)
	valLabel = sample['label'].long().to(device)
	out = net(valImg)
	out = F.log_softmax(out, dim=1)
	pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
	pre = cm[pre_label]
	pre1 = Image.fromarray(pre)
	pre1.save(dir + str(i) + '.png')
	print('Done')
	break

import matplotlib.pyplot as plt

print(sample['label'][0].shape)
plt.imshow(sample['label'][0])
plt.show()