import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from evalution_segmentaion import eval_semantic_segmentation
from FCN import FCN
from datasets import ImageDatasets
import torchvision.transforms as transforms

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

TRAIN_ROOT = './data/train'
TRAIN_LABEL = './data/train_labels'
VAL_ROOT = './data/val'
VAL_LABEL = './data/val_labels'
CLASS_DICT_PATH = './data/class_dict.csv'

crop_size = (224, 352)

trans_image = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

Image_train = ImageDatasets([TRAIN_ROOT, TRAIN_LABEL, CLASS_DICT_PATH], crop_size, trans_image)
Image_val = ImageDatasets([VAL_ROOT, VAL_LABEL, CLASS_DICT_PATH], crop_size, trans_image)

DataLoader_train=DataLoader(Image_train,batch_size=4,shuffle=True,num_workers=1)
DataLoader_val=DataLoader(Image_val,batch_size=4,shuffle=True,num_workers=1)


fcn=FCN(12)
fcn=fcn.to(device)
criterion=nn.NLLLoss().to(device)
optimizer=optim.Adam(fcn.parameters(),lr=1e-4)

def train(model,num_epochs):
    best=[0]
    net=model.train()
    for epoch in range(num_epochs):

        # if epoch % 10 == 0 and epoch != 0:
        #     for group in optimizer.param_groups:
        #         group["lr"] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0

        for i, sample in enumerate(DataLoader_train):
            img_data = Variable(sample["img"].to(device))
            img_label = Variable(sample["label"].to(device))

            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metric = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metric["mean_class_accuracy"]
            print(eval_metric["mean_class_accuracy"])
            train_miou += eval_metric["miou"]
            train_class_acc += eval_metric["class_accuracy"]

            if(i+1)%10==0:
                print("Epoch {}/{} Batch: {}/{} Batch_loss: {:.8f}"
                      .format(epoch,num_epochs,i+1,len(DataLoader_train),loss.item()))

        metric_description='|train Acc: {:.5f}| Train Mean UI| :{:.5f}\n Train_class_acc:{}'.format(
            train_acc/len(DataLoader_train),
            train_miou/len(DataLoader_train),
            train_class_acc/len(DataLoader_train),
        )

        print(metric_description)

        if max(best)<=train_miou/len(DataLoader_train):
            best.append(train_miou/len(DataLoader_train))
            torch.save(net.state_dict(),'{}.path'.format(epoch))

if __name__ == '__main__':
    train(fcn,20)
