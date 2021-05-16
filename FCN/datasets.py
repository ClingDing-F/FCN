import pandas as pd
import os
import torch
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class Label_Processor:
    def __init__(self,file_path):

        self.colormap = self.read_color_map(file_path)

        self.color_label_pixels = self.encode_label_pix(self.colormap)

    def read_color_map(self,file_path):
        label_color=pd.read_csv(file_path,sep=',')
        color_map=[]
        # print(label_color.shape)
        for i in range(label_color.shape[0]):
            tmp=label_color.iloc[i]
            color_map.append([tmp['r'],tmp['g'],tmp['b']])
        return color_map

    def encode_label_pix(self,colormap):

        label_pixels=np.zeros(256**3)

        for i,color_m in enumerate(colormap):
            label_pixels[(color_m[0]*256+color_m[1])*256+color_m[2]]=i

        return label_pixels

    def encode_label_img(self,img):
        img=np.array(img,dtype='int32')
        idx = (img[:, :, 0] * 256 + img[:, :, 1]) * 256 + img[:, :, 2]
        return np.array(self.color_label_pixels[idx], dtype='int64')

class ImageDatasets(Dataset):
    def __init__(self,file_path=[] ,crop_size=None,transform=None):
        if(len(file_path)!=3):
            raise ValueError("file path should include images and labels path")
        self.img_path=file_path[0]
        self.label_path=file_path[1]
        self.colormap_path=file_path[2]

        self.label_processor=Label_Processor(self.colormap_path)

        self.imgs=self.read_file(self.img_path)
        self.labels=self.read_file(self.label_path)
        self.crop_size=crop_size
        self.transform=transform

    def __getitem__(self, index):
        img=self.imgs[index]
        label=self.labels[index]
        img=Image.open(img)
        label=Image.open(label).convert('RGB')

        img, label=self.crop_center_image(img,label,self.crop_size)

        img, label=self.img_transform(img,label)

        result = {'img':img,"label":label}
        return  result


    def __len__(self):
        return len(self.imgs)


    def read_file(self,path):
        files_list=os.listdir(path)
        file_path_list=[os.path.join(path,img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def crop_center_image(self,img,label,crop_size):
        img=F.center_crop(img,crop_size)
        label = F.center_crop(label, crop_size)
        return img,label

    def img_transform(self,img,label):

        img=self.transform(img)

        label=Image.fromarray(np.array(label).astype('uint8'))

        label=self.label_processor.encode_label_img(label)

        label=torch.from_numpy(label)

        return img,label


if __name__ == '__main__':

    TRAIN_ROOT = './data/train'
    TRAIN_LABEL = './data/train_labels'
    VAL_ROOT = './data/val'
    VAL_LABEL = './data/val_labels'
    TEST_ROOT = './data/test'
    TEST_LABEL = './data/test_labels'
    CLASS_DICT_PATH = './data/class_dict.csv'
    crop_size = (200, 200)

    trans_image = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    Image_train = ImageDatasets([TRAIN_ROOT, TRAIN_LABEL, CLASS_DICT_PATH], crop_size,trans_image)

    Image_val = ImageDatasets([VAL_ROOT, VAL_LABEL, CLASS_DICT_PATH], crop_size, trans_image)

    Image_test = ImageDatasets([TEST_ROOT, TEST_LABEL, CLASS_DICT_PATH], crop_size, trans_image)

    print(len(Image_train))
    print(len(Image_val))
    print(len(Image_test))


