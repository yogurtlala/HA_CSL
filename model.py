import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from einops import rearrange 
import numpy as np


class  Attention_Layer(nn.Module):
    def __init__(self, ):
        super(Attention_Layer, self).__init__()


    def forward(self, x, w, bias, gamma):
        out = x.contiguous().view(x.size(0)*x.size(1), x.size(2))##outshape bs*pn,512

        out_f = F.linear(out, w, bias)##out_f bs*pn,2

        out = out_f.view(x.size(0),x.size(1), out_f.size(1))##bs,pn,2

        out= torch.sqrt((out**2).sum(2))##对patch的概率进行相加再开平方

        alpha_01 = out /out.sum(1, keepdim=True).expand_as(out)#分母相当于L2范式,有点像算每个patch占整个图片的权重
        #alpha_01 shape为16,64    2维上相加为1
        alpha_01 = F.relu(alpha_01- 0.8/float(gamma))##relu小于0为0，稀疏性，进一步如果权重太小则设置为0
        
        alpha_01 = alpha_01/alpha_01.sum(1, keepdim=True).expand_as(alpha_01)##进行归一化

        alpha = torch.unsqueeze(alpha_01, dim=2)##bs,pn,1
        out = alpha.expand_as(x)*x 

        return out, out_f, alpha_01

class LossAttention2D1(nn.Module):
    def __init__(self,patchsize = 16):
        super(LossAttention2D1,self).__init__()
        self.patchsize = patchsize
        self.model = monai.networks.nets.resnet18(spatial_dims=2, n_input_channels=1, num_classes=2)
        self.feature_extractor_part1 = nn.Sequential(*list(self.model.children())[:-2])
        self.Pool = nn.AdaptiveMaxPool2d(1)
        self.attention1 = Attention_Layer()
        self.attention2 = Attention_Layer()
        self.attention3 = Attention_Layer()
        self.linear1 = nn.Linear(512,2)
        self.linear2 = nn.Linear(512,2)
        self.linear3 = nn.Linear(64,1)

        
    def forward(self,x,flag=1):
        """MP方法：先得到图像的权重，再得到每个Patch的权重。是一个使用了LossBasedAttention的输入为2D的网络
        具体包括
        1）通过resnet得到每张每个patch的特征，即d,n,c(d张-n个patch-c个通道)
        2）对每张的patch进行sparse_attention,得到d,n
        3）通过对注意力权重做注意力，得到d,1(图像权重)
        4）用dx1得到融合后的特征，即n,c(patch权重)，再做lossbasedatt
        输入为图片,shape为 b(batch_size),c(channel为1),d(深度为d,也就是d张图像),h(长),w(宽)
        输出为 z_y(shape为bs,2), z_out_c(bs,pn,2), z_alpha(bs,pn)
        """
        b,c,d,h,w = x.shape        
        p = self.patchsize  
        out = rearrange(x,'b c d (h p1) (w p2) -> (b d h w) c p1 p2', p1=p, p2=p) ##输出为bsxdxpn，c,h,w (pn为patch的数量)

        ##1）NETWORK 得到每张每个patch的特征
        H = self.feature_extractor_part1(out)
        H = self.Pool(H)##bsxdxpn，c 
        H = H.view(b,d, (int)(H.shape[0]/(b*d)), -1)##H_shape为bs,d,pn,c

        ##2）得到每张每个patch的attention_weight
        flag = (int)(H.shape[2])##flag代表patch_num    
        ##对（nxm，c）做attention 
        out, out_c, alpha = self.attention1(H.view(b,d*flag,-1), self.linear1.weight, self.linear1.bias, (int)(d*flag))##alpha为bs，dxpn
        
        ##对（n,c）做attention
        # oCut, out_c, alpha = self.attention1(H.view(b*d,flag,-1), self.linear1.weight, self.linear1.bias, (int)(flag))##alpha为bsxd,pn

        ##3）用每张每个patch的attention_weight，得到每个patch的attention_weight
        alpha = alpha.view((int)(b),(int)(d),-1)##bs,d,pn

        # ##2018attention
        # alpha2 = self.attention2(alpha)##输出为bs,d,1 ##其实就是个mlp
        # alpha2 = torch.softmax(alpha2,dim = 1).transpose(1,2)##输出为bs,1,d
        # patch_weight = torch.matmul(alpha2,alpha)##输出为bs,1,pn

        # ##mlp
        # alpha2 = self.attention3(alpha)##输出为bs,d,1
        # alpha2 = torch.softmax(alpha2,dim = 1)
        # patch_weight = alpha2.transpose(1,2)##输出为bs,1,d
        # print(patch_weight)

        ##sparse attention
        _,_,alpha2 = self.attention2(alpha, self.linear3.weight, self.linear3.bias, d)#输出为bs,d,1
        patch_weight = torch.unsqueeze(alpha2,dim=1)
        print(patch_weight)


        ##4）对融合后的特征做lossbasedattention
        patch_weight = torch.repeat_interleave(patch_weight,flag,dim=1).view((int)(b*flag),1,-1)##bsxn,1,d
        z = torch.matmul(patch_weight, H.transpose(1,2).contiguous().view(b*flag,d,-1)).view((int)(b),flag,-1)##bs,pn,c
        z_out, z_out_c, z_alpha = self.attention3(z, self.linear2.weight, self.linear2.bias, flag)

        ###bag_featureA
        z_out = z_out.sum(1)
        z_out = z_out.view(z_out.size(0),-1)
        z_y = self.linear2(z_out)##y_shape bs,2
        return  z_y, z_out_c, z_alpha
    
class MixMoudle2(nn.Module):
    def __init__(self,patchsize = 16):
        super(MixMoudle2,self).__init__()
        self.patchsize = patchsize
        ###########2d################
        self.model2d = monai.networks.nets.resnet18(spatial_dims=2, n_input_channels=1, num_classes=2)##backbone为resnet18
        self.feature_extractor_part12d = nn.Sequential(*list(self.model2d.children())[:-2])
        self.Pool2d = nn.AdaptiveMaxPool2d(1)##全局平均池化，输出都为512，1，1，1
        self.attention12d = Attention_Layer()
        self.attention22d = Attention_Layer()
        self.attention32d = Attention_Layer()
        self.linear12d = nn.Linear(512,2)
        self.linear22d = nn.Linear(512,2)
        self.linear32d = nn.Linear(64,1)

        ########3d#######################
        self.model3d = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)##backbone为resnet18
        self.feature_extractor_part13d = nn.Sequential(*list(self.model3d.children())[:-2])
        self.Pool3d = nn.AdaptiveMaxPool3d(1)##全局平均池化，输出都为512，1，1，1
        self.attention3d = Attention_Layer()
        self.linear3d = nn.Linear(512,2)


    def forward(self,x,flag=1):
        """MixMoudle2融合了3DLossAtt和2DLossAtt
        输入为图片,shape为 b(batch_size),c(channel为1),d(深度为d,也就是d张图像),h(长),w(宽)
        输出包括
            2D模型输出:z_y2d(shape为bs,2), z_out_c2d(bs,pn,2), z_alpha2d(bs,pn)
            3D模型输出:y3d(bs,2), out_c3d(bs,pn,2), alpha3d (bs,pn)
        """
        ##--------------------------------------2d------------------------------------------------
        b,c,d,h,w = x.shape        
        p = self.patchsize  
        out2d = rearrange(x,'b c d (h p1) (w p2) -> (b d h w) c p1 p2', p1=p, p2=p) ##输出为bsxdxpn，c,h,w

        ##1）NETWORK 得到每张每个patch的特征
        H2d = self.feature_extractor_part12d(out2d)
        H2d = self.Pool2d(H2d)##bsxdxpn，c 
        H2d = H2d.view(b,d, (int)(H2d.shape[0]/(b*d)), -1)##H_shape为bs,d,pn,c

        ##2）得到每张每个patch的attention_weight
        flag = (int)(H2d.shape[2])##flag代表patch_num    
        ##对（nxm，c）做attention 
        out2d, out_c2d, alpha2d = self.attention12d(H2d.view(b,d*flag,-1), self.linear12d.weight, self.linear12d.bias, (int)(d*flag))##alpha为bs，dxpn
        
        ##3）用每张每个patch的attention_weight，得到每个patch的attention_weight
        alpha2d = alpha2d.view((int)(b),(int)(d),-1)##bs,d,pn
        ##sparse attention
        _,_,alpha22d = self.attention22d(alpha2d, self.linear32d.weight, self.linear32d.bias, d)#输出为bs,d,1
        patch_weight2d = torch.unsqueeze(alpha22d,dim=1)
     


        ##4）对融合后的特征做lossbasedattention
        ###得到融合后的图像
        patch_weight2d = torch.repeat_interleave(patch_weight2d,flag,dim=1).view((int)(b*flag),1,-1)##bsxpn,1,d
        z2d = torch.matmul(patch_weight2d, H2d.transpose(1,2).contiguous().view(b*flag,d,-1)).view((int)(b),flag,-1)##bs,pn,c
        ###sparse attention
        z_out2d, z_out_c2d, z_alpha2d = self.attention32d(z2d, self.linear22d.weight, self.linear22d.bias, flag)##z_alpha为bs,pn

        ###bag_feature
        z_out2d = z_out2d.sum(1)
        z_out2d = z_out2d.view(z_out2d.size(0),-1)###z_out为bs,c
        z_y2d = self.linear22d(z_out2d)##y_shape bs,2



        ##---------------------------------3d--------------------------------------
        b,c,d,h,w = x.shape        
        ##3D
        out3d = rearrange(x,'b c d (h p1) (w p2) -> (b h w) c p1 p2 d', p1=p, p2=p) 

        # ##NETWORK
        H3d = self.feature_extractor_part13d(out3d)
        H3d = self.Pool3d(H3d)##bsxdxpn,512

        ###loss attention
        flag = (int)(H3d.shape[0]/b)
        H3d = H3d.view((int)(b), (int)(H3d.shape[0]/b), -1)##bs,pn,512
        ##通过lossatt得到重要的patch,需要输入L-1层的特征，和L层参数
        out3d, out_c3d, alpha3d = self.attention3d(H3d, self.linear3d.weight, self.linear3d.bias, flag)
        ###out :bs,pn,512   out_c bs*pn,2

        out3d = out3d.sum(1)
        out3d = out3d.view(out3d.size(0),-1)##bs, 512
        y3d = self.linear3d(out3d)##y_shape bs,2

        ###return
        return  z_y2d, z_out_c2d, z_alpha2d,y3d, out_c3d, alpha3d 

class MixMoudle_Add(nn.Module):
    def __init__(self,patchsize = 16,reduction_ratio = 16):
        super(MixMoudle_Add,self).__init__()
        self.patchsize = patchsize
        ###########2d################
        self.model2d = monai.networks.nets.resnet18(spatial_dims=2, n_input_channels=1, num_classes=2)
        self.feature_extractor_part12d = nn.Sequential(*list(self.model2d.children())[:-2])
        self.Pool2d = nn.AdaptiveMaxPool2d(1)
        self.attention12d = Attention_Layer()
        self.attention22d = Attention_Layer()
        self.attention32d = Attention_Layer()
        self.linear12d = nn.Linear(512,2)
        # self.linear22d = nn.Linear(512,2)
        self.linear32d = nn.Linear(64,1)

        ########3d#######################
        self.model3d = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
        self.feature_extractor_part13d = nn.Sequential(*list(self.model3d.children())[:-2])
        self.Pool3d = nn.AdaptiveMaxPool3d(1)
        self.attention3d = Attention_Layer()
        # self.linear3d = nn.Linear(512,2)

        #######MIX######################
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )##concat classifier
        self.classifier2 = nn.Linear(512,2)##add classifier
        self.BN = nn.BatchNorm1d(512)
        ##channel attention得到注意力权重
        self.CA_mlp = nn.Sequential(
            nn.Linear(512,512 // reduction_ratio),##nn.Conv1d和nn.linear计算结果相同
            nn.ReLU(),
            nn.Linear(512 // reduction_ratio, 512),
            nn.Softmax(dim=1),##or softmax

        )
        self.CA_mlp_all = nn.Sequential(
            nn.Linear(1024,1024 // reduction_ratio),
            nn.ReLU(),
            nn.Linear(1024 // reduction_ratio, 1024),
            nn.Softmax(dim=1),

        )


    def forward(self,x,shared=0,CA_case=0,flag=1):
        """MixMoudle_ADD融合了3DLossAtt和2DLossAtt
        输入为图片,shape为b,c,h,w,d
        输出包括
            2D模型输出:z_y2d(shape为bs,2), z_out_c2d(bs,pn,2), z_alpha2d(bs,pn)
            3D模型输出:y3d(bs,2), out_c3d(bs,pn,2), alpha3d (bs,pn)
            融合后输出:output(bs,2)
        """
        ##-------------------------2D--------------------------------------------------
        b,c,d,h,w = x.shape        
        p = self.patchsize  
        out2d = rearrange(x,'b c d (h p1) (w p2) -> (b d h w) c p1 p2', p1=p, p2=p) ##输出为bsxdxpn，c,h,w

        ##1）NETWORK 得到每张每个patch的特征
        H2d = self.feature_extractor_part12d(out2d)
        H2d = self.Pool2d(H2d)##bsxdxpn，c 
        H2d = H2d.view(b,d, (int)(H2d.shape[0]/(b*d)), -1)##H_shape为bs,d,pn,c

        ##2）得到每张每个patch的attention_weight
        flag = (int)(H2d.shape[2])##flag代表patch_num  
        if shared == 0:
            ##对（nxm，c）做attention 
            out2d, out_c2d, alpha2d = self.attention12d(H2d.view(b,d*flag,-1), self.linear12d.weight, self.linear12d.bias, (int)(d*flag))##alpha为bs，dxpn
        elif shared == 1:
            out2d, out_c2d, alpha2d = self.attention12d(H2d.view(b,d*flag,-1), self.classifier2.weight, self.classifier2.bias, (int)(d*flag))##alpha为bs，dxpn

        ##3）用每张每个patch的attention_weight，得到每个patch的attention_weight
        alpha2d = alpha2d.view((int)(b),(int)(d),-1)##bs,d,pn
        ##sparse attention
        _,_,alpha22d = self.attention22d(alpha2d, self.linear32d.weight, self.linear32d.bias, d)#输出为bs,d,1
        patch_weight2d = torch.unsqueeze(alpha22d,dim=1)
        # print(patch_weight2d)


        ##4）对融合后的特征做lossbasedattention
        patch_weight2d = torch.repeat_interleave(patch_weight2d,flag,dim=1).view((int)(b*flag),1,-1)##bsxn,1,d
        z2d = torch.matmul(patch_weight2d, H2d.transpose(1,2).contiguous().view(b*flag,d,-1)).view((int)(b),flag,-1)##bs,pn,c
        z_out2d, z_out_c2d, z_alpha2d = self.attention32d(z2d, self.classifier2.weight, self.classifier2.bias, flag)

        ###bag_featureA
        z_out2d = z_out2d.sum(1)
        z_out2d = z_out2d.view(z_out2d.size(0),-1)##z_out shape: bs,c
        z_y2d = self.classifier2(z_out2d)##y_shape bs,2



        ##-------------------------------3D----------------------------------------
        b,c,d,h,w = x.shape        
        ##3D
        out3d = rearrange(x,'b c d (h p1) (w p2) -> (b h w) c p1 p2 d', p1=p, p2=p) 

        # ##NETWORK
        H3d = self.feature_extractor_part13d(out3d)
        H3d = self.Pool3d(H3d)##bsxdxpatch_number,512##得到每一张每个patch的weight

        ###loss attention
        flag = (int)(H3d.shape[0]/b)##flg为多少个patch
        H3d = H3d.view((int)(b), (int)(H3d.shape[0]/b), -1)##bs,pn,512
        ##通过lossatt得到重要的patch,需要输入L-1层的特征，和L层参数
        out3d, out_c3d, alpha3d = self.attention3d(H3d, self.classifier2.weight, self.classifier2.bias, flag)
        ##out为bs,pn,1(每个patch乘权重后的结果) ，out_c是H直接映射到2维的结果
        ###out :bs,pn,512   out_c bs*pn,2

        out3d = out3d.sum(1)
        out3d = out3d.view(out3d.size(0),-1)##bs, 512
        y3d = self.classifier2(out3d)##y_shape bs,2

        ##--------------------------ADD OR CONCAT-------------------------------------------
        z_out2d = self.BN(z_out2d)
        out3d = self.BN(out3d)
        if CA_case == 0:
            ##合并特征
            feature = torch.add(z_out2d,out3d)##bs,c
            z_y = self.classifier2(feature)##bs,2
           

        elif CA_case == 1:

            ##得到每个通道的注意力权重
            z_out2d_weight = self.CA_mlp(z_out2d)
            out3d_weight = self.CA_mlp(out3d)

            ##应用注意力权重到原始特征，加强或减弱不同通道的贡献
            z_out2d = torch.mul(z_out2d_weight, z_out2d)
            out3d = torch.mul(out3d_weight,out3d)
            ##合并特征
            feature = torch.add(z_out2d,out3d)##bs,c
            z_y = self.classifier2(feature)##bs,2
        
        elif CA_case == 2:
            feature = torch.add(z_out2d,out3d)##bs,c
            feature_weight = self.CA_mlp(feature)
            feature = torch.mul(feature_weight, feature)
            z_y = self.classifier2(feature)##bs,2
        elif CA_case == 3:
            out = torch.cat((z_out2d,out3d),dim=1)
            out_weight = self.CA_mlp_all(out)
            out = torch.mul(out_weight,out)
            z_out2d, out3d =torch.split(out,512,dim=1)
            ##合并特征
            feature = torch.add(z_out2d,out3d)##bs,c
            z_y = self.classifier2(feature)##bs,2








    


        return  z_y2d, z_out_c2d, z_alpha2d,y3d, out_c3d, alpha3d,z_y