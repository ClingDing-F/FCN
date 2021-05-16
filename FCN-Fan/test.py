import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from evalution_segmentaion import eval_semantic_segmentation
from datasets import ImageDatasets
from FCN import FCN
import torchvision.transforms as transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 4
miou_list = [0]

TEST_ROOT = './data/test'
TEST_LABEL = './data/test_labels'
CLASS_DICT_PATH = './data/class_dict.csv'
crop_size = (224, 352)

trans_image = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

Image_test = ImageDatasets([TEST_ROOT, TEST_LABEL, CLASS_DICT_PATH], crop_size, trans_image)
DataLoader_test = DataLoader(Image_test, batch_size=2, shuffle=True, num_workers=0)

net = FCN(12)
net.eval()
net.to(device)
net.load_state_dict(torch.load('./15.path'))

train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0

for i, sample in enumerate(DataLoader_test):
	data = Variable(sample['img']).to(device)
	label = Variable(sample['label']).to(device)
	out = net(data)
	out = F.log_softmax(out, dim=1)

	pre_label = out.max(dim=1)[1].data.cpu().numpy()
	pre_label = [i for i in pre_label]

	true_label = label.data.cpu().numpy()
	true_label = [i for i in true_label]

	eval_metrix = eval_semantic_segmentation(pre_label, true_label)
	train_acc = eval_metrix['mean_class_accuracy'] + train_acc
	train_miou = eval_metrix['miou'] + train_miou
	train_mpa = eval_metrix['pixel_accuracy'] + train_mpa
	if len(eval_metrix['class_accuracy']) < 12:
		eval_metrix['class_accuracy'] = 0
		train_class_acc = train_class_acc + eval_metrix['class_accuracy']
		error += 1
	else:
		train_class_acc = train_class_acc + eval_metrix['class_accuracy']

	print(eval_metrix['class_accuracy'], '================', i)


epoch_str = ('test_acc :{:.5f} ,test_miou:{:.5f}, test_mpa:{:.5f}, test_class_acc :{:}'
			 .format(train_acc /(len(DataLoader_test)-error),train_miou/(len(DataLoader_test)-error),
					 train_mpa/(len(DataLoader_test)-error),train_class_acc/(len(DataLoader_test)-error)))

if train_miou/(len(DataLoader_test)-error) > max(miou_list):
	miou_list.append(train_miou/(len(DataLoader_test)-error))
	print(epoch_str+'==========last')
