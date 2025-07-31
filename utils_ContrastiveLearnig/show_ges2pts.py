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
from visualization import *
#showpoints(np.random.randn(2048,3), c1 = np.random.uniform(0,1,size = (2048)))

parser = argparse.ArgumentParser()

choice="Mug"

parser.add_argument('--model', type=str, default='model_Contratstive_Parts2Gesture', help='model path')
parser.add_argument('--idx', type=int, default=300, help='model index')
parser.add_argument('--dataset', type=str, default='neuralnet_dataset', help='dataset path')

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    split='train',
    data_augmentation=False)

_, seglabel, hand_set,label, batch_weight = d[opt.idx]

hand=np.split(hand_set,2,axis=0)
hand_l = hand[0]-hand[0][0]
hand_r = hand[1]-hand[1][0]
hand_l=hand_l.reshape(1,69)
hand_r=hand_r.reshape(1,69)

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


min_logit_sk_l=0
min_logit_parts_l=0
min_logit_sk_r=0
min_logit_parts_r=0

p_l=np.array([])
p_r=np.array([])
ges_l_path=""
ges_r_path=""
l_move = np.array([])
r_move = np.array([])


for pts_csv in os.listdir("neuralnet_dataset/search/pts"):
    #print(pts_csv)
    pts_path = os.path.join("neuralnet_dataset/search/pts" , pts_csv)
    point_set=np.array(pd.read_csv(pts_path,header=None)).astype(np.float32)
    choice = np.random.choice(point_set.shape[0], 2048, replace=True)
    #resample
    point_set = point_set[choice, :]
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #scale

    point_set = torch.from_numpy(point_set)

    #partsseg
    point = point_set.transpose(1, 0).contiguous()

    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _, all_feat= classifier(point)
    pred_choice = pred.data.max(2)[1].cpu()

    pl=np.array([])
    pr=np.array([])
    parts_l_list=np.array([])
    parts_r_list=np.array([])

    pred_choice=pred_choice[0]

    if (np.count_nonzero(pred_choice==1) != 0) and (np.count_nonzero(pred_choice==2)!=0):
        for j in range(2048):
            if pred_choice[j]==2:
                parts_l_list=np.append(parts_l_list,point_set[j])
            if pred_choice[j]==1:
                parts_r_list=np.append(parts_r_list,point_set[j])

        while len(parts_l_list)<=(3 * 256):
            add_list=parts_l_list*1.01
            parts_l_list=np.append(parts_l_list,add_list)
        while len(parts_r_list)<=(3 * 256):
            add_list=parts_r_list*1.01
            parts_r_list=np.append(parts_r_list,add_list)
        #sampling
        parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3)             
        parts_r_list=parts_r_list.reshape(int(len(parts_r_list)/3),3)   
        choice = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True) 
        choice = np.random.choice(int(parts_r_list.shape[0]), 256, replace=True)
        pl=np.append(pl,parts_l_list[choice,:])          
        pr=np.append(pr,parts_r_list[choice,:])

        pl=pl.reshape(1,256,3).astype(np.float32)
        pr=pr.reshape(1,256,3).astype(np.float32)
        #指を0,0,0にするから移動させよう。(後に使う)
        pl_move = np.expand_dims(np.mean(pl, axis = 1), 0)
        pr_move = np.expand_dims(np.mean(pr, axis = 1), 0)

        pl = pl - pl_move
        pr = pr - pr_move
        pl=torch.from_numpy(pl)
        pr=torch.from_numpy(pr)

        pl=pl.transpose(2,1)
        pr=pr.transpose(2,1)

        parts_l, parts_r, all_feat =  pl, pr, all_feat
        #hand_l, hand_r = hand_l.cuda(), hand_r.cuda()
        logit_per_sk_l, logit_per_parts_l, sk_feat_l, parts_feat_l = sk_parts_classifier(hand_l, parts_l, all_feat)
        logit_per_sk_r, logit_per_parts_r, sk_feat_r, parts_feat_r = sk_parts_classifier(hand_r, parts_r, all_feat)

        if logit_per_sk_l > min_logit_sk_l:
            pf_l = parts_feat_l
            p_l = parts_l
            l_move=pl_move
            min_logit_sk_l = logit_per_sk_l
            parts_name_l = pts_csv

        if  logit_per_sk_r >min_logit_sk_r:
            pf_r = parts_feat_r
            p_r = parts_r
            r_move = pr_move
            min_logit_sk_r = logit_per_sk_r
            parts_name_r = pts_csv

print("---推定結果---")
print("parts_l:",parts_name_l, "sim:", min_logit_sk_l)
print("parts_r:",parts_name_r, "sim:", min_logit_sk_r)

min_logit_per_p=0
pred_pts_csv=""
pred_pts=np.array([])

#最良探索

for pts_csv in os.listdir("neuralnet_dataset/search/pts"):
    pts_path = os.path.join("neuralnet_dataset/search/pts", pts_csv)
    point_set=np.array(pd.read_csv(pts_path,header=None)).astype(np.float32)
    choice = np.random.choice(point_set.shape[0], 2048, replace=True)
    #resample
    point_set = point_set[choice, :]
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #scale
    point_set = torch.from_numpy(point_set)
    #partsseg
    point = point_set.transpose(1, 0).contiguous()
    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _, all_feat = classifier(point)
    
    logit_per_p, logit_per_pts = p2pts_classifier(pf_l, pf_r, all_feat)

    if logit_per_p > min_logit_per_p:
        pred_pts_csv=pts_path
        min_logit_per_p = logit_per_p
        pred_pts=point_set
    print(logit_per_p,pts_csv)
    

ges_l = hand_l.cpu().numpy().reshape(23,3) 
ges_r = hand_r.cpu().numpy().reshape(23,3) 
pred_parts_l = p_l[0].transpose(0,1).contiguous().cpu().numpy()+l_move
pred_parts_r = p_r[0].transpose(0,1).contiguous().cpu().numpy()+r_move

#parts - pts

#matplotlibで表示しよう。
#pointset, seglabel ,predchoice
print("===========")
print("推定結果")
print("全体の形状は",pred_pts_csv)

fig, ax = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax.set_title('input_ges', fontsize=10) 

ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.set_zlim(-1.5,1.5)

fig1, ax1 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax1.set_title('pred_parts', fontsize=10) 
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.set_zlim(-1,1)

fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax2.set_title('pred_pts', fontsize=10) 
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.set_zlim(-1,1)

fig3, ax3 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax3.set_title('output', fontsize=10) 
ax3.set_xlim(-1.5,1.5)
ax3.set_ylim(-1.5,1.5)
ax3.set_zlim(-1,1)

#pointset, seglabel ,predchoice, ges_l, ges_r
#input
drawhand(hand=hand[0],color="red",ax=ax)
drawhand(hand=hand[1],color="blue",ax=ax)
#predict
#parts l,r
drawparts(pred_parts_l, ax=ax1, parts="left" )
drawparts(pred_parts_r, ax=ax1, parts="right")

point_set=pred_pts.numpy()
drawparts(point_set,ax=ax2,parts="")

#output

#ges and pts
drawhand(hand=hand[0],color="red",ax=ax3)
drawhand(hand=hand[1],color="blue",ax=ax3)
drawparts(point_set,ax=ax3,parts="")

plt.show()



