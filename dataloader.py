"""Pytorch dataset object that loads MNIST dataset as bags."""
import time
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import os
import SimpleITK as sitk
from monai.transforms import LoadImage, Randomizable, apply_transform
from trans import resample,GetRoi,GetRoi2,GetRoi3,GetRoi_topo,GetRoi_topo_pvp
import seaborn as sns
from skimage import transform
from monai.transforms import (
    AsChannelFirst,
    RandGaussianNoise,
    AddChannel,
    EnsureChannelFirst,
    RandSpatialCropSamples,
    Compose,
    RandRotate90,
    Resize,
    RandFlip,
    ScaleIntensity,
    SpatialCropD,
    ToTensor,
    RandRotate,
    ToNumpy,
    RandBiasField,
    RandCropByPosNegLabel,
    RandCropByLabelClasses,
    RandSpatialCrop,
    RandAffine,
    Rand3DElastic,
    RandAdjustContrast,
    NormalizeIntensity,
    Spacing,
)

##加上mask,输出的是经过pad和resize后的ROI图像##对比一下MIL有 什么作用
class BagsROI2(data_utils.Dataset):
    def __init__(self,sample_name ,sample_labels,transform):
        self.sample_name = sample_name
        self.sample_labels = sample_labels
        
        self.transform = transform
        self.bags_list, self.label = self._create_bags(sample_name,sample_labels)##generate label and imge
   
         
    def _create_bags(self,sample_name, sample_labels, 
            shuffle_flag=True):
        """
            datasets for img roi
            return
                img tesnor of [bs,1,100,100,100]
        """
       
        bags_list = []
        labels_list = []
        
        for i_iter in range(len(sample_name)):

            #read image
            try:
                img_dir = sample_name[i_iter]
                
                ##get roi and resize to 100x100x100
                img = GetRoi2(img_dir)
                # print("img",img.shape)
                
                bags_list.append(img)
                labels_list.append(sample_labels[i_iter])   
            except:
                continue
        return bags_list,labels_list  

    def __len__(self):
        
        return len(self.label)

    def __getitem__(self, index):

        # trans_t0 = time.time()

        bag = apply_transform(self.transform,self.bags_list[index])
        label = self.label[index]
        
        # trans_t1 = time.time()
        # print(trans_t1 - trans_t0)


        return bag, label         


if __name__ == "__main__":
    train_transforms = Compose(
                        [
                            
                            ScaleIntensity(),
                            RandRotate(),##旋转
                            RandFlip(),##翻转
                            ToNumpy()
                        ]
                    )
    normal_scan_paths = []
    abnormal_scan_paths = []
    normal_topo_paths = []
    abnormal_topo_paths = []
    topo_root = '/home/pst/code/topo/matrix/old'##放topo的位置
    root = "/home/pst/HCC_DATA/AP-feature/old" #如何输入两次的数据
    normal_scan_paths,abnormal_scan_paths,normal_topo_paths,abnormal_topo_paths = get(root,topo_root,normal_scan_paths,abnormal_scan_paths,normal_topo_paths,abnormal_topo_paths)
    # topo_root = '/home/pst/code/topo/matrix/new'##放topo的位置
    # root = "/home/pst/HCC_DATA/AP-feature/new" #如何输入两次的数据
    # normal_scan_paths,abnormal_scan_paths,normal_topo_paths,abnormal_topo_paths = get(root,topo_root,normal_scan_paths,abnormal_scan_paths,normal_topo_paths,abnormal_topo_paths)

    fold_size = 0.1
    normal_len = (int)(fold_size*len(normal_scan_paths))
    abnormal_len = (int)(fold_size*len(abnormal_scan_paths))
    scan_paths = normal_scan_paths[:normal_len] + abnormal_scan_paths[:abnormal_len]
    topo_paths = normal_topo_paths[:normal_len]  + abnormal_topo_paths[:abnormal_len]
    y = np.concatenate(([1] * normal_len, [0] * abnormal_len), axis=0)
    y = np.array(y, dtype=np.int64)

    train_sample_name = scan_paths
    train_sample_labels = y
    train_loader = data_utils.DataLoader(MRIDataset(train_sample_name , train_sample_labels,train_transforms),batch_size = 4)

    len_bag_list_train = []

    for batch_idx, (bag, label) in enumerate(train_loader):
        ##统计一个包中instance的数量
        print(bag.shape)
        # print(int(bag.squeeze(0).size()[0]))
        # len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
   
        
    
