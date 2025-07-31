from __future__ import print_function
#from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetDenseCls , ContrastiveNet
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

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=400, help='model index')
parser.add_argument('--dataset', type=str, default='neuralnet_dataset_unity', help='dataset path')

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    split='search',
    data_augmentation=False)

point_set, seglabel, hand_set,label, batch_weight = d[opt.idx]

#model load 
"----partseg----"
state_dict = torch.load("model_Contratstive_Parts2Gesture/pointnet_model_loss_total_best.pth",weights_only=False)
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval()

"----pts2ges----"
state_dict_contrastive = torch.load("model_Contratstive_Parts2Gesture/contrastive_model_loss_partseg_best.pth",weights_only=False)
sk_parts_classifier = ContrastiveNet()
sk_parts_classifier.load_state_dict(state_dict_contrastive)
sk_parts_classifier.eval()

point = point_set.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))

pred, _, _, all_feat= classifier(point)
pred_choice = pred.data.max(2)[1].cpu()

print("AAAAAAA")



pl=np.array([])
pr=np.array([])


parts_l_list=np.array([])
parts_r_list=np.array([])
#pred_choice = pred_choice.cpu().data.numpy()
print("推測  label 1 ,2 ,0:",np.count_nonzero(pred_choice==1),np.count_nonzero(pred_choice==2),np.count_nonzero(pred_choice==0))
print("答え  label 1 ,2 ,0:",np.count_nonzero(seglabel==1),np.count_nonzero(seglabel==2),np.count_nonzero(seglabel==0))

print(pred_choice.shape)
pred_choice=pred_choice[0]
print(pred_choice.shape)

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
pr=np.append(pr,parts_r_list[choice,:])
pl=np.append(pl,parts_l_list[choice,:])

pl=pl.reshape(1,256,3).astype(np.float32)
pr=pr.reshape(1,256,3).astype(np.float32)

pl_move = np.expand_dims(np.mean(pl, axis = 1), 0)
pr_move = np.expand_dims(np.mean(pr, axis = 1), 0)

pl = pl - pl_move
pr = pr - pr_move

pl=torch.from_numpy(pl)
pr=torch.from_numpy(pr)
pl=pl.transpose(2,1)
pr=pr.transpose(2,1)

parts_l, parts_r, all_feat =  pl, pr, all_feat

min_logit_sk_l=0
min_logit_parts_l=0
min_logit_sk_r=0
min_logit_parts_r=0

ges_l=np.array([])
ges_r=np.array([])

ges_l_path=""
ges_r_path=""

for hands_csv in os.listdir("neuralnet_dataset/search/hands"):
    hands_path = os.path.join("neuralnet_dataset/search/hands" , hands_csv)
    hand_set=np.array(pd.read_csv(hands_path,header=None)).astype(np.float32)
    hand=np.split(hand_set,2,axis=0)
    hand_l = hand[0]-hand[0][0]
    hand_r = hand[1]-hand[1][0]
    hand_l=hand_l.reshape(1,69)
    hand_r=hand_r.reshape(1,69)
    hand_l , hand_r = torch.from_numpy(hand_l) , torch.from_numpy(hand_r) 
    #hand_l, hand_r = hand_l.cuda(), hand_r.cuda()
    logit_per_sk_l, logit_per_parts_l, sk_feat_l, parts_feat_l = sk_parts_classifier(hand_l, parts_l, all_feat)
    logit_per_sk_r, logit_per_parts_r, sk_feat_r, parts_feat_r = sk_parts_classifier(hand_r, parts_r, all_feat)

    
    if logit_per_parts_l > min_logit_parts_l:
        ges_l = hand_l
        min_logit_parts_l = logit_per_parts_l
        ges_l_path = hands_path

    if logit_per_parts_r > min_logit_parts_r:
        ges_r = hand_r
        min_logit_parts_r = logit_per_parts_r
        ges_r_path = hands_path

print("---推定結果---")
print("ges_l:",ges_l_path, "sim:", min_logit_sk_l)
print("ges_r:",ges_r_path, "sim:", min_logit_sk_r)

ges_l = ges_l.cpu().numpy().reshape(23,3) 
ges_l = ges_l - np.expand_dims(np.mean(ges_l, axis = 0), 0) + pl_move[0]

ges_r = ges_r.cpu().numpy().reshape(23,3) 
ges_r = ges_r - np.expand_dims(np.mean(ges_r, axis = 0), 0) + pr_move[0]

#matplotlibで表示しよう。

fig, ax = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax.set_title('ans', fontsize=20) 
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

fig1, ax1 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax1.set_title('partseg', fontsize=20) 
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.set_zlim(-1,1)

fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax2.set_title('parts', fontsize=20) 
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.set_zlim(-1,1)

fig7, ax7 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax7.set_title('input', fontsize=20) 
ax7.set_xlim(-1,1)
ax7.set_ylim(-1,1)
ax7.set_zlim(-1,1)


fig3, ax3 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax3.set_title('output_ges', fontsize=20) 
ax3.set_xlim(-1.5,1.5)
ax3.set_ylim(-1.5,1.5)
ax3.set_zlim(-1,1)

fig4, ax4 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax4.set_title('parts_l - ges_l', fontsize=20) 
ax4.set_xlim(-1.5,1.5)
ax4.set_ylim(-1.5,1.5)
ax4.set_zlim(-1,1)

fig5, ax5 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax5.set_title('parts_r - ges_r', fontsize=20) 
ax5.set_xlim(-1.5,1.5)
ax5.set_ylim(-1.5,1.5)
ax5.set_zlim(-1,1)

fig6, ax6 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax6.set_title('pts-ges', fontsize=20) 
ax6.set_xlim(-1.5,1.5)
ax6.set_ylim(-1.5,1.5)
ax6.set_zlim(-1,1)




#pointset, seglabel ,predchoice, ges_l, ges_r
#ans
point_set=point_set.numpy()
drawpts(point_set,seglabel,ax=ax)
drawparts(point_set, ax=ax7, parts="")
#predict
drawpts(point_set,pred_choice,ax=ax1)
#parts l,r
drawparts(parts_l_list, ax=ax2, parts="left" )
drawparts(parts_r_list, ax=ax2, parts="right")

#output
drawhand(hand=ges_l,color="red",ax=ax3)
drawhand(hand=ges_r,color="blue",ax=ax3)

#parts_l and ges_l
drawhand(hand=ges_l,color="red",ax=ax4)
drawparts(parts_l_list, ax=ax4, parts="left" )

#parts_l and ges_l
drawhand(hand=ges_r,color="blue",ax=ax5)
drawparts(parts_r_list, ax=ax5, parts="right")

#out pts and ges
drawhand(hand=ges_l,color="red",ax=ax6)
drawhand(hand=ges_r,color="blue",ax=ax6)
drawparts(point_set,ax=ax6,parts="")

plt.show()



