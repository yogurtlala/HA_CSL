from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch.nn.functional as F
from scipy import interp
import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' ##GPUCPU处理数据是否并行
from dataloader import BagsROI2
from model import LossAttention2D1,MixMoudle2,MixMoudle_Add
from weight_loss import CrossEntropyLoss as CE
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import monai
from metric_helpper import comp_specificity,comp_sensitivity
from trans import myshuffle,GetRoi2
from monai.transforms import (
    RandGaussianNoise,
    AddChannel,
    Compose,
    RandFlip,
    ScaleIntensity,
    ScaleIntensityRangePercentiles,
    ToTensor,
    RandRotate,
    RandAffine,
    RandAdjustContrast,
    NormalizeIntensity,
    RandGaussianSmooth,
    RandScaleIntensity,
)
import torch, gc
import os
import SimpleITK as sitk
import random
import xlwt
from sklearn.metrics import roc_auc_score, f1_score, precision_score,roc_curve,auc
from monai.utils import set_determinism
from sklearn.model_selection import KFold,StratifiedKFold
from collections import Counter
import torch.nn as nn
from sklearn import metrics
from datetime import datetime
# Settings
f = xlwt.Workbook('encoding = utf-8')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:248"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='MRI_Attention')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=2022, metavar='S',
                    help='random seed')
parser.add_argument('--file_path', type=str, default='./0912_CA_ADD', metavar='file_path',
                    help='path of tensorboard')
parser.add_argument('--case', type=str, default='ADD', metavar='case',
                    help='resnet or MP(2DLossBased) or MIX or ADD')
parser.add_argument('--choice', type=str, default='base', metavar='choice',
                    help=' base or distill or sim or mse')
parser.add_argument('--bs', type=int, default='16', metavar='bs',
                    help='batch_size')
parser.add_argument('--Temp', type=float, default='0.7', metavar='bs',
                    help='Temp')#调整平滑度
parser.add_argument('--sigma', type=float, default='10', metavar='sigma',
                    help='sigma of rampup function')
parser.add_argument('--shared', type=int, default='0', metavar='shared',
                    help='shared classifier')
parser.add_argument('--rr', type=int, default='16', metavar='rr',
                    help='reduction_ratio')
parser.add_argument('--CA_case', type=int, default='3', metavar='rr',
                    help='0 OR 1 OR 2 OR 3')
args = parser.parse_args()
writer = SummaryWriter(args.file_path)
set_determinism(args.seed)##设置随机种子
fileouter = open('{}/log.txt'.format(args.file_path),'a')###文件输出
print('Load Train and Test Set') 
print("the current time is",datetime.now(),file=fileouter)
for arg in vars(args):
    print(arg,':',getattr(args,arg),file=fileouter)
best_model_root = args.file_path +'/model'
##Transforms
train_transforms = Compose(
    [
        AddChannel(),
        ScaleIntensityRangePercentiles(0,99.99,0,1),####HU值
        RandScaleIntensity(0.1),#v=v*(1+factor)
        RandRotate(),##旋转
        RandFlip(),##翻转
        RandAffine(),##仿射变换，效果包括缩放、旋转、平移、反射，错切（shear）、直线放射后还是直线，但是感觉错切会对病灶产生形变
        RandGaussianNoise(),
        RandGaussianSmooth(),
        RandAdjustContrast(),
        ToTensor(),
        NormalizeIntensity(),##归一化
    ]
)
test_transforms = Compose(
    [
        AddChannel(),
        ScaleIntensity(),
        ToTensor(),
        NormalizeIntensity(),
        
    ]
)

##rampup
def rampup(global_step, rampup_length=40):
    if global_step <rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)

#knowledge distillation loss
def distillation(y,teacher_scores,labels,T,alpha):
    #学生logits的softmax,再log
    p = F.log_softmax(y/T,dim=1)
    #老师logits的softmax-t
    q = F.softmax(teacher_scores/T,dim=1)
    #学生老师之间的kl散度之类的
    l_kl = F.kl_div(p,q,size_average=False)*(T**2)/y.shape[0]
    return l_kl
   

# 获取数据
def get(root,normal_scan_paths,abnormal_scan_paths):
    path_list = os.listdir(root)
    path_list.sort(key=lambda x:int(x[:-6]))
    for file in path_list:
        ##检查是否能读
        try:
            feature_root = "/data/pst/hbp/HBP-feature/HBP-feature/"+root[-3:]
            path = os.path.join(feature_root, file[:-4]+'.nii.gz')
            GetRoi2(path)
            if file.endswith('1.nii'):
                normal_scan_paths.append(os.path.join(feature_root, file[:-4]+'.nii.gz'))
            else:
                abnormal_scan_paths.append(os.path.join(feature_root, file[:-4]+'.nii.gz'))
        except:
            continue
    return normal_scan_paths,abnormal_scan_paths
def generate_path_label(root):
        
    normal_scan_paths = []
    abnormal_scan_paths = []
    normal_scan_paths,abnormal_scan_paths = get(root,normal_scan_paths,abnormal_scan_paths)

    ##label
    scan_paths = normal_scan_paths + abnormal_scan_paths
    labels = np.concatenate(([1] * len(normal_scan_paths), [0]*len(abnormal_scan_paths)), axis=0)
    labels = np.array(labels, dtype=np.int64)
    return scan_paths,labels 

root = "/data/pst/hbp/HBP-ROI/HBP-ROI/old"
scan_paths , labels = generate_path_label(root)
root = "/data/pst/hbp/HBP-ROI/HBP-ROI/new"
scan_paths2 , labels2 = generate_path_label(root)
scan_paths = np.append(scan_paths,scan_paths2)
labels = np.append(labels,labels2)
topo_paths = []
scan_paths,labels,topo_paths = myshuffle(scan_paths,labels,topo_paths)##shuffle


### Main 五折交叉验证
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
acc_list, auc_list, spe_list, sen_list ,f1_list, epoch_list,pre_list = [],[],[],[],[],[],[]##用于保存每一折的结果
k=0 ##记录当前epoch数
for train_index, val_index in kf.split(scan_paths,labels):
    k = k+1
    print(k,"epoch",file = fileouter)
    train_scan_paths,train_labels = np.array(scan_paths)[train_index],labels[train_index]
    valid_scan_paths,valid_labels = np.array(scan_paths)[val_index],labels[val_index]  

    ##dataloader
    train_ds = BagsROI2(train_scan_paths ,train_labels,train_transforms)
    train_loader = data_utils.DataLoader(train_ds,batch_size = args.bs)
    test_loader = data_utils.DataLoader(BagsROI2(valid_scan_paths , valid_labels,test_transforms),batch_size = args.bs)


    ##model
    if args.case == 'resnet':
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
    elif args.case == 'MP':
        model = LossAttention2D1().to(device)##2DLossAtt
    elif args.case == 'MIX':
        model = MixMoudle2().to(device)##distill or mse or simlarity or 平均加权
    elif args.case == 'ADD':
        model = MixMoudle_Add(reduction_ratio=args.rr).to(device)###add or concat


    ### Model Setting
    T=args.Temp##定义distill损失中的温度
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    mseloss= nn.MSELoss()##定义mseloss方法下的损失
    cosloss = nn.CosineEmbeddingLoss()##定义simlarity方法下的损失
    weight_criterion = CE(aggregate='mean')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.333, 
                patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)



    ##for n epoch
    best_acc,best_auc,best_f1,best_sen,best_spe,best_pre,best_val_labels_list,best_val_score_list ,best_val_outputs_list= -1,-1,-1,-1,-1,-1,-1,-1,-1
    for epoch in range(1, args.epochs + 1):
        ##设置每个epoch下的u_w
        if  args.case == 'MP'or args.case == 'MIX'or args.case =='ADD':
            rampup_value = rampup(epoch)
            if epoch==0:
                u_w = 0
            else:
                u_w = 0.1*rampup_value
                u_w_distill = args.sigma*rampup_value
            u_w = torch.autograd.Variable(torch.FloatTensor([u_w]).cuda(), requires_grad=False) 
            u_w_distill = torch.autograd.Variable(torch.FloatTensor([u_w_distill]).cuda(), requires_grad=False) 

        ##---------------------------------------------Train------------------------------------------------------------
        model.train()
        train_loss ,  train_acc  = 0.,0.
        l = 0
        ##for n batch
        for batch_idx, (data, label) in enumerate(train_loader):
            bag_label = label.to(device)
            data = data.to(device)
            optimizer.zero_grad() # reset gradients

            if args.case =='resnet' :
                ##3D RESNET
                outputs = model(data)
                loss = criterion(outputs, bag_label.long())
                train_loss += loss.item() 


            elif args.case == 'MP':
                ##2D LossBasedAttention
                outputs, y_c, alpha = model(data)##Class:LossAttention2D1
                loss_1 = criterion(outputs, bag_label)
                loss_2 = weight_criterion(y_c, bag_label.repeat(64,1).permute(1,0).contiguous().view(-1), weights=alpha.view(-1))
                loss = loss_1+ u_w*loss_2/outputs.size(0)
                train_loss += loss.data[0] ##loss相加 

            elif args.case == 'ADD':
                ## Mix Model by add or concat
                outputs_2d, y_c_2d, alpha_2d,outputs_3d, y_c_3d, alpha_3d, outputs = model(data,args.shared,args.CA_case)##Class:MixMoudle_Add
                loss_1 = criterion(outputs, bag_label)
                loss_2_2d = weight_criterion(y_c_2d, bag_label.repeat(64,1).permute(1,0).contiguous().view(-1), weights=alpha_2d.view(-1))
                loss_2_3d = weight_criterion(y_c_3d, bag_label.repeat(64,1).permute(1,0).contiguous().view(-1), weights=alpha_3d.view(-1))
                loss_2 = loss_2_2d + loss_2_3d
                ##不同情况下计算损失
                if args.choice == 'base':
                    loss = loss_1+ u_w*(loss_2)/outputs.size(0)
                elif args.choice == 'distill':
                    sim_loss = distillation(alpha_3d,alpha_2d,bag_label,T,0.1)##distillloss
                    loss = 1*(loss_1+ u_w*loss_2/outputs.size(0)) +u_w_distill*sim_loss
                elif args.choice == 'mse':
                    sim_loss = mseloss(alpha_3d,alpha_2d)
                    loss = 0.9*(loss_1+ u_w*(loss_2)/(outputs.size(0))) +0.1*sim_loss##mseloss
                elif args.choice == 'sim':
                    loss_flag = torch.ones(alpha_2d.shape[0]).to(device)
                    sim_loss = cosloss(alpha_2d,alpha_3d,loss_flag)##similarity loss
                    loss = 0.9*(loss_1+ u_w*(loss_2)/(outputs.size(0))) + 0.1*sim_loss
                train_loss += loss.data[0] ##loss相加 

            elif args.case == 'MIX':
                ## 对融合后的模型加上知识蒸馏损失、MSE、Similarity
                outputs_2d, y_c_2d, alpha_2d,outputs_3d, y_c_3d, alpha_3d = model(data)##Class:MixMoudle2
                ##平均加权
                outputs = 0.5*outputs_2d + 0.5*outputs_3d
                y_c = 0.5*y_c_2d + 0.5*y_c_3d
                alpha = 0.5*alpha_2d +0.5*alpha_3d
                ##calculate bagloss and instance_loss
                loss_1 = criterion(outputs, bag_label)
                loss_2 = weight_criterion(y_c, bag_label.repeat(64,1).permute(1,0).contiguous().view(-1), weights=alpha.view(-1))

                ##不同情况下计算损失
                if args.choice == 'base':
                    loss = loss_1+ u_w*(loss_2)/outputs.size(0)
                elif args.choice == 'distill':
                    distillation_loss = distillation(alpha_3d,alpha_2d,bag_label,T,0.1)##distillloss
                    loss = 1*(loss_1+ u_w*loss_2/outputs.size(0)) +0.2*distillation_loss
                elif args.choice == 'mse':
                    sim_loss = mseloss(alpha_3d,alpha_2d)
                    loss = 0.9*(loss_1+ u_w*(loss_2)/(outputs.size(0))) +0.1*sim_loss##mseloss
                elif args.choice == 'sim':
                    loss_flag = torch.ones(alpha_2d.shape[0]).to(device)
                    sim_loss = cosloss(alpha_2d,alpha_3d,loss_flag)##similarity loss
                    loss = 0.9*(loss_1+ u_w*(loss_2)/(outputs.size(0))) + 0.1*sim_loss


                train_loss += loss.data[0] ##loss相加 
                

            value = torch.eq(outputs.argmax(dim=1), bag_label)
            train_acc+= value.sum().item()##正确次数
            l += len(value)##总次数
            loss.backward()
            # step
            optimizer.step()##根据loss反向梯度进行参数更新

        
        
        # calculate loss and error for epoch
        train_loss /= l
        train_acc /= l
        writer.add_scalar("{}/train/train_loss".format(k), train_loss, epoch)
        writer.add_scalar("{}/train/train_acc".format(k), train_acc, epoch)
        print('\nEpoch: {}, l:{}, Loss: {:.4f}, Train acc: {:.4f},train_size:{}'.format(epoch,l, train_loss, train_acc, l),file = fileouter)



        ##------------------------------------------model eval----------------------------------------------------------------------
        model.eval()
        test_loss ,test_acc, test_sim_loss = 0.,0. ,0.
        l = 0
        val_outputs_list , val_labels_list,val_score_list  = list(), list() , list()
        for batch_idx, (data, label) in enumerate(test_loader):
            """每个batch下, 输入是data和label,输出是累加的loss,acc,list
            """
            bag_label = label.to(device)
            data = data.to(device)
            with torch.no_grad():

                if args.case =='resnet' :
                    outputs = model(data)
                    loss = criterion(outputs, bag_label.long())
                    test_loss += loss.item()
                  

                elif args.case == 'MP':
                    outputs, y_c, alpha = model(data)##LossAttention2D1
                    loss_1 = criterion(outputs, bag_label)
                    loss_2 = weight_criterion(y_c, bag_label.repeat(64,1).permute(1,0).contiguous().view(-1), weights=alpha.view(-1))
                    loss = loss_1+ u_w*loss_2/outputs.size(0)
                    test_loss += loss.data[0] 

                elif args.case == 'ADD':
                    outputs_2d, y_c_2d, alpha_2d,outputs_3d, y_c_3d, alpha_3d, outputs = model(data,args.shared,args.CA_case)##Model为MixMoudle_Add
                    loss_1 = criterion(outputs, bag_label)
                    loss_2_2d = weight_criterion(y_c_2d, bag_label.repeat(64,1).permute(1,0).contiguous().view(-1), weights=alpha_2d.view(-1))
                    loss_2_3d = weight_criterion(y_c_3d, bag_label.repeat(64,1).permute(1,0).contiguous().view(-1), weights=alpha_3d.view(-1))
                    loss_2 = loss_2_2d + loss_2_3d
                    ##不同情况下计算损失
                    if args.choice == 'base':
                        sim_loss = torch.zeros(1,dtype=torch.float).to(device)
                        loss = loss_1+ u_w*(loss_2)/outputs.size(0)
                        
                    elif args.choice == 'distill':
                        sim_loss = distillation(alpha_3d,alpha_2d,bag_label,T,0.1)##distillloss
                        loss = 1*(loss_1+ u_w*loss_2/outputs.size(0)) +u_w_distill*sim_loss
                    elif args.choice == 'mse':
                        sim_loss = mseloss(alpha_3d,alpha_2d)
                        loss = 0.9*(loss_1+ u_w*(loss_2)/(outputs.size(0))) +0.1*sim_loss##mseloss
                    elif args.choice == 'sim':
                        loss_flag = torch.ones(alpha_2d.shape[0]).to(device)
                        sim_loss = cosloss(alpha_2d,alpha_3d,loss_flag)##similarity loss
                        loss = 0.9*(loss_1+ u_w*(loss_2)/(outputs.size(0))) + 0.1*sim_loss
                    test_loss += loss.data[0] ##loss相加 
                    test_sim_loss += sim_loss.item()
                
                elif args.case == 'MIX':
                    outputs_2d, y_c_2d, alpha_2d,outputs_3d, y_c_3d, alpha_3d = model(data)##Model为MixMoudle2
                    outputs = 0.5*outputs_2d + 0.5*outputs_3d
                    y_c = 0.5*y_c_2d + 0.5*y_c_3d
                    alpha = 0.5*alpha_2d +0.5*alpha_3d
                    loss_1 = criterion(outputs, bag_label)
                    loss_2 = weight_criterion(y_c, bag_label.repeat(64,1).permute(1,0).contiguous().view(-1), weights=alpha.view(-1))

                    if args.choice == 'base':
                        loss = loss_1+ u_w*(loss_2)/outputs.size(0) 
                    elif args.choice == 'distill':
                        distillation_loss = distillation(alpha_3d,alpha_2d,bag_label,T,0.1)
                        loss = 1*(loss_1+ u_w*loss_2/outputs.size(0)) +0.2*distillation_loss
                    elif args.choice == 'mse':
                        sim_loss = mseloss(alpha_3d,alpha_2d)
                        loss = 0.9*(loss_1+ u_w*(loss_2)/(outputs.size(0))) +0.1*sim_loss
                    elif args.choice == 'sim':
                        loss_flag = torch.ones(alpha_2d.shape[0]).to(device)
                        sim_loss = cosloss(alpha_2d,alpha_3d,loss_flag)
                        loss = 0.9*(loss_1+ u_w*(loss_2)/(outputs.size(0))) + 0.1*sim_loss
                    test_loss += loss.data[0] ##loss相加 
                    
                value = torch.eq(outputs.argmax(dim=1), bag_label)
                test_acc += value.sum().item()
                l += len(value)
                val_labels_list.extend(bag_label.cpu().numpy().tolist())
                val_outputs_list.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
                val_score_list.extend(torch.softmax(outputs,dim=1)[:,1].cpu().numpy().tolist())

     
        #求的每个epoch的结果
        test_loss /= l
        test_acc /= l
        test_sim_loss /= l
        ##学习率调整
        scheduler.step(test_loss)     
        ##写入tensorboard
        val_f1 = f1_score(val_labels_list,val_outputs_list)
        val_auc = roc_auc_score(val_labels_list,val_score_list) 
        val_spe = comp_specificity(val_labels_list, val_outputs_list)
        val_sen =comp_sensitivity(val_labels_list, val_outputs_list)
        val_precision =  precision_score(val_labels_list,val_outputs_list)
        writer.add_scalar( "{}/test/sensitivity".format(k), val_sen, epoch )
        writer.add_scalar("{}/test/specificity".format(k),val_spe,epoch)
        writer.add_scalar("{}/test/f1-score".format(k), val_f1, epoch )
        writer.add_scalar("{}/test/precision".format(k),val_precision, epoch )
        writer.add_scalar("{}/test/auc".format(k), val_auc, epoch )
        writer.add_scalar("{}/test/test_loss".format(k), test_loss, epoch)
        writer.add_scalar("{}/test/test_sim_loss".format(k), test_sim_loss, epoch)
        writer.add_scalar("{}/test/test_acc".format(k), test_acc, epoch)
        writer.add_scalar("{}/lr/lr".format(k), optimizer.param_groups[0]["lr"], epoch)
        print(" lr",optimizer.param_groups[0]["lr"],file = fileouter)
        print('Test Set, Loss: {:.4f}, Test acc: {:.4f}, auc: {:.4f}, f1-score: {:.4f}, sensitivity: {:.4f}, specification: {:.4f}'.format(
            test_loss, test_acc,val_auc,val_f1,val_sen,val_spe),file = fileouter)
    
  

        ##保存最好结果和模型
        if test_acc > best_acc:
            print("--------best acc--------",file = fileouter)
            best_acc = test_acc
            best_f1 =  val_f1
            best_auc = val_auc
            best_sen = val_sen
            best_spe = val_spe
            best_pre = val_precision
            best_epoch = epoch
            best_val_labels_list = val_labels_list
            best_val_score_list = val_score_list
            best_val_outputs_list = val_outputs_list

            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()},"{}{}.pth".format(best_model_root,k) )
            print("saved new best metric model")

    ##保存每折结果
    acc_list.append(best_acc)
    spe_list.append(best_spe)
    sen_list.append(best_sen)
    pre_list.append(best_pre)
    auc_list.append(best_auc)
    f1_list.append(best_f1)
    epoch_list.append(best_epoch)
    print('\nTest Set, ACC: {:.4f} , auc: {:.4f}  ,f1: {:.4f} , sen: {:.4f}, spe: {:.4f},   Epoch:{}'.format(best_acc,best_auc,best_f1,best_sen,best_spe,best_epoch),file = fileouter)   



        
##输出五折平均的结果
print("\n-------------------5fold_result-----------")
print("mean_auc",np.mean(auc_list),np.std(auc_list),file = fileouter)
print(auc_list,file = fileouter)
print("mean_acc",np.mean(acc_list),np.std(acc_list),file = fileouter)
print(acc_list,file = fileouter)
print("mean_f1",np.mean(f1_list),np.std(f1_list),file = fileouter)
print(f1_list,file = fileouter)
print("mean_sen",np.mean(sen_list),np.std(sen_list),file = fileouter)
print(sen_list,file = fileouter)
print("mean_pre",np.mean(pre_list),np.std(pre_list),file = fileouter)
print(pre_list,file = fileouter)
print("mean_spe",np.mean(spe_list),np.std(spe_list),file = fileouter)
print(spe_list,file = fileouter)
print("mean_epoch",np.mean(epoch_list),np.std(epoch_list),file = fileouter)
print(epoch_list,file = fileouter)

writer.close()
fileouter.close()
    

