from __future__ import print_function
#from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetDenseCls , ContrastiveNet, PartsToPtsNet
import matplotlib.pyplot as plt
import sys
import os
import cv2
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset
#showpoints(np.random.randn(2048,3), c1 = np.random.uniform(0,1,size = (2048)))

"""
cv2でコサイン類似度の行列を0→1でグラデーションしたimgを作成したい。
すべてに対して行うか?ミニバッチで行うか?
ミニバッチで行うなら、各ピクセルサイズを3?
"""

"""
img_out_size=dataset_size * batchsize
img_out=np.zeros(img_out_size,img_out_size)
"""


parser = argparse.ArgumentParser()

choice="Mug"

parser.add_argument('--model', type=str, default='C:/Users/yokada/Desktop/new/pointnet.pytorch/segqq/seg_model_'+choice+'_loss.pth', help='model path')
parser.add_argument('--samplesize', type=int, default=16, help='input batch size')
parser.add_argument('--dataset', type=str, default='neuralnet_dataset_unity', help='dataset path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    split='val',
    data_augmentation=False)

database =  ShapeNetDataset(
    root=opt.dataset,
    split='search',
    data_augmentation=False)

dloader = DataLoader(d,batch_size=10,shuffle=False)

#databaseloader = DataLoader(database,batch_size=len(os.listdir(os.path.join(opt.dataset,"search/pts"))),shuffle=False)
databaseloader = DataLoader(database,batch_size=len(os.listdir(os.path.join(opt.dataset,"search/pts"))),shuffle=False)

#model load 

"----partseg----"
state_dict = torch.load("model_Contratstive_Parts2Gesture/pointnet_model_loss_total_best.pth",weights_only=False)
#print( state_dict['conv4.weight'].size()[0])
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval()

"----pts2ges----"
state_dict_contrastive = torch.load("model_Contratstive_Parts2Gesture/contrastive_model_loss_partseg_best.pth",weights_only=False)
sk_parts_classifier = ContrastiveNet()
sk_parts_classifier.load_state_dict(state_dict_contrastive)
sk_parts_classifier.eval()
"----parts2pts----"
state_dict_contrastive = torch.load("model_Contratstive_Parts2Gesture/parts2pts_model_loss_total_best.pth",weights_only=False)
p2pts_classifier = PartsToPtsNet()
p2pts_classifier.load_state_dict(state_dict_contrastive)
p2pts_classifier.eval()


pts,seg, hands, label, batch_w = next(iter(dloader))
pts_base, seg_base, hands_base, label_base, batch_w_base = next(iter(databaseloader))

arr_mIoU_partseg=np.array([])
arr_name=np.array([])

for pts_csv in os.listdir("neuralnet_dataset_unity/search/pts"):
    pts_path = os.path.join("neuralnet_dataset_unity/search/pts" , pts_csv)
    point_set=np.array(pd.read_csv(pts_path,header=None)).astype(np.float32)
    choice = np.random.choice(point_set.shape[0], 2048, replace=True)
    #resample
    point_set = point_set[choice, :]
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #scale

    point_set = torch.from_numpy(point_set)

    seg_csv=pts_csv.split(".")[0]+"_label.csv"
    seg_path=os.path.join("neuralnet_dataset_unity/search/label" , seg_csv)
    seg_label=np.array(pd.read_csv(seg_path,header=None)).astype(np.int64)
    seg_label = seg_label[choice,:]
    seg_label=torch.from_numpy(seg_label)
    seg_label=seg_label.transpose(1,0)

    #partsseg
    point = point_set.transpose(1, 0).contiguous()

    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _, all_feat= classifier(point)
    pred_choice = pred.data.max(2)[1]
    print(seg_label.shape,pred_choice.shape)
    mIoU_partseg=torch.sum(torch.eq(seg_label,pred_choice))/2048
    print(mIoU_partseg)
    arr_mIoU_partseg=np.append(arr_mIoU_partseg,mIoU_partseg.item())
    arr_name=np.append(arr_name,pts_csv[:2])
    
arr_out=np.stack((arr_mIoU_partseg,arr_name),1)
df=pd.DataFrame(arr_out)
df.to_csv("mIoU_partseg.csv",header=None,index=None)

classes = np.unique(arr_out[:, 1])
# 各クラスの平均を計算
averages = {}
for cls in classes:
    # クラスに対応する行を取得
    values = arr_out[arr_out[:, 1] == cls, 0].astype(np.float32)
    # 平均を計算
    averages[cls] = np.mean(values)

# 結果を表示
for cls, avg in averages.items():
    print(f"{cls} の平均: {avg}")




    
    
    