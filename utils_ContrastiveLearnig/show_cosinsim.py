from __future__ import print_function
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

"""
cv2でコサイン類似度の行列を0→1でグラデーションしたimgを作成する。

"""



parser = argparse.ArgumentParser()

choice="Mug"

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--samplesize', type=int, default=16, help='input batch size')
parser.add_argument('--dataset', type=str, default='neuralnet_dataset_unity', help='dataset path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--class_choice', type=str, default=choice, help='class choice')

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

#すべてのジェスチャーの中のどれと一番マッチするか
def parts2ges_all(p,g,seg):
    outarr_p_size=p.shape[0]
    outarr_g_size=g.shape[0]
    points =p.transpose(2, 1).contiguous()
    pred, _, _,all_feat = classifier(points)
    pred = pred.view(-1, 3)

    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().data.numpy()
    # pred_choice 1 :left,  pred_choice 2 : right
    
    points=points.transpose(1, 2).cpu().data.numpy()
    pred_np=pred_np.reshape(outarr_p_size,2048,1)
    pl=np.array([])
    pr=np.array([])
    for batch in range(outarr_p_size):
        count=0
        parts_l_list=np.array([])
        parts_r_list=np.array([])
        #print(batch,np.count_nonzero(pred_np[batch]==1),np.count_nonzero(pred_np[batch]==2) )
        target_l=pred_np
        target_r=pred_np
        if np.count_nonzero(pred_np[batch]==2)<=10:    
            target_l=seg
            print("セグメント失敗",batch)
        else:
            target_l=pred_np
        if np.count_nonzero(pred_np[batch]==1)<=10:
            target_r=seg
            print("セグメント失敗",batch)
        else:
            target_r=pred_np

        for j in range(2048):
            if target_l[batch][j]==2:
                parts_l_list=np.append(parts_l_list,points[batch][j])
            if target_r[batch][j]==1:
                parts_r_list=np.append(parts_r_list,points[batch][j])

        while len(parts_l_list)<=(3 * 256):
            count+=1
            add_list=parts_l_list*1.01
            parts_l_list=np.append(parts_l_list,add_list)

        while len(parts_r_list)<=(3 * 256):
            count+=1
            add_list=parts_r_list*1.01
            parts_r_list=np.append(parts_r_list,add_list)
        
        parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3)             
        choice = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True)                
        pl=np.append(pl,parts_l_list[choice,:])
        parts_r_list=parts_r_list.reshape(int(len(parts_r_list)/3),3)             
        choice = np.random.choice(int(parts_r_list.shape[0]), 256, replace=True)                
        pr=np.append(pr,parts_r_list[choice,:])
        pl = pl - np.expand_dims(np.mean(pl, axis = 0), 0)
        pr = pr - np.expand_dims(np.mean(pr, axis = 0), 0)
    pl=pl.reshape(outarr_p_size,256,3).astype(np.float32)
    pr=pr.reshape(outarr_p_size,256,3).astype(np.float32)

    pl=torch.from_numpy(pl)
    pr=torch.from_numpy(pr)
    pl=pl.transpose(2,1)
    pr=pr.transpose(2,1)
    #------------------------------------------------

    hand=np.split(g,2,axis=1)
    hand_l=hand[0]
    hand_r=hand[1]
    #手首を0に
    for k in range(outarr_g_size):
        hand_l[k] = hand_l[k] - hand_l[k][0]
        hand_r[k] = hand_r[k] - hand_r[k][0]

    hand_l=hand_l.reshape(outarr_g_size,69)
    hand_r=hand_r.reshape(outarr_g_size,69)
    sim_pl_gl, _, _, parts_l_feat =sk_parts_classifier(hand_l, pl, all_feat)
    sim_pr_gr, _, _, parts_r_feat =sk_parts_classifier(hand_r, pr, all_feat)

    sim_pl_gr, _, _, parts_l_feat =sk_parts_classifier(hand_l, pr, all_feat)
    sim_pr_gl, _, _, parts_r_feat =sk_parts_classifier(hand_r, pl, all_feat)

    sim_pl_per_g = torch.cat((sim_pl_gl,sim_pl_gr),dim=1)
    sim_pr_per_g = torch.cat((sim_pr_gl,sim_pr_gr),dim=1)
    
    sim_parts_per_ges = torch.cat((sim_pl_per_g, sim_pr_per_g),dim=0).transpose(1,0)
    sim_parts_per_ges = sim_parts_per_ges.detach().numpy()
    print("AAAAAAAAAAAAAAAAA")
    mat=sim_parts_per_ges.shape[1]//sim_parts_per_ges.shape[0]
    if mat==0:
        mat=1
    print("===========")

    img=np.zeros((mat*sim_parts_per_ges.shape[0],sim_parts_per_ges.shape[1]))

    for k in range(mat):
        for h in range(sim_parts_per_ges.shape[0]):
            for w in range(sim_parts_per_ges.shape[1]):
                img[sim_parts_per_ges.shape[0]*k+h][w]= sim_parts_per_ges[h][w]

    block_size = 3
    image_blocked = np.kron(img, np.ones((block_size, block_size)))
    image = image_blocked

    return image

#全部と全部
#image=parts2ges_all(pts,hands)

def parts2ges(p,g,seg):
    outarr_p_size=p.shape[0]
    outarr_g_size=g.shape[0]
    points =p.transpose(2, 1).contiguous()
    pred, _, _,all_feat = classifier(points)
    pred = pred.view(-1, 3)

    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().data.numpy()
    # pred_choice 1 :left,  pred_choice 2 : right
    
    points=points.transpose(1, 2).cpu().data.numpy()
    pred_np=pred_np.reshape(outarr_p_size,2048,1)
    pl=np.array([])
    pr=np.array([])
    for batch in range(outarr_p_size):
        count=0
        parts_l_list=np.array([])
        parts_r_list=np.array([])
        #print(batch,np.count_nonzero(pred_np[batch]==1),np.count_nonzero(pred_np[batch]==2) )
        target_l=pred_np
        target_r=pred_np
        if np.count_nonzero(pred_np[batch]==2)<=10:    
            target_l=seg
            print("セグメント失敗",batch)
        else:
            target_l=pred_np
        if np.count_nonzero(pred_np[batch]==1)<=10:
            target_r=seg
            print("セグメント失敗",batch)
        else:
            target_r=pred_np

        for j in range(2048):
            if target_l[batch][j]==2:
                parts_l_list=np.append(parts_l_list,points[batch][j])
            if target_r[batch][j]==1:
                parts_r_list=np.append(parts_r_list,points[batch][j])

        while len(parts_l_list)<=(3 * 256):
            count+=1
            add_list=parts_l_list*1.01
            parts_l_list=np.append(parts_l_list,add_list)
        while len(parts_r_list)<=(3 * 256):
            count+=1
            add_list=parts_r_list*1.01
            parts_r_list=np.append(parts_r_list,add_list)

        parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3)             
        choice = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True)                
        pl=np.append(pl,parts_l_list[choice,:])
        parts_r_list=parts_r_list.reshape(int(len(parts_r_list)/3),3)             
        choice = np.random.choice(int(parts_r_list.shape[0]), 256, replace=True)                
        pr=np.append(pr,parts_r_list[choice,:])
        pl = pl - np.expand_dims(np.mean(pl, axis = 0), 0)
        pr = pr - np.expand_dims(np.mean(pr, axis = 0), 0)
    pl=pl.reshape(outarr_p_size,256,3).astype(np.float32)
    pr=pr.reshape(outarr_p_size,256,3).astype(np.float32)

    pl=torch.from_numpy(pl)
    pr=torch.from_numpy(pr)
    pl=pl.transpose(2,1)
    pr=pr.transpose(2,1)

    hand=np.split(g,2,axis=1)
    hand_l=hand[0]
    hand_r=hand[1]
    #手首を0に
    for k in range(outarr_g_size):
        hand_l[k] = hand_l[k] - hand_l[k][0]
        hand_r[k] = hand_r[k] - hand_r[k][0]

    hand_l=hand_l.reshape(outarr_g_size,69)
    hand_r=hand_r.reshape(outarr_g_size,69)
    sim_pl_gl, _, _, parts_l_feat =sk_parts_classifier(hand_l, pl, all_feat)
    sim_pr_gr, _, _, parts_r_feat =sk_parts_classifier(hand_r, pr, all_feat)

    sim_parts_per_g_l = sim_pl_gl.transpose(1,0)
    sim_parts_per_ges_l= sim_parts_per_g_l.detach().numpy()
    print(sim_parts_per_ges_l)
    sim_parts_per_g_r = sim_pr_gr.transpose(1,0)
    sim_parts_per_ges_r= sim_parts_per_g_r.detach().numpy()
    print("===========")

    img_l=np.zeros((sim_parts_per_ges_l.shape[0],sim_parts_per_ges_l.shape[1]))
    img_r=np.zeros((sim_parts_per_ges_r.shape[0],sim_parts_per_ges_r.shape[1]))

    for h in range(sim_parts_per_g_l.shape[0]):
        for w in range(sim_parts_per_g_l.shape[1]):
            img_l[h][w]= sim_parts_per_g_l[h][w]

    for h in range(sim_parts_per_g_r.shape[0]):
        for w in range(sim_parts_per_g_r.shape[1]):
            img_r[h][w]= sim_parts_per_g_r[h][w]

    #parts2pts
    sim_parts_per_pts, _ = p2pts_classifier(parts_l_feat, parts_r_feat, all_feat)
    sim_parts_per_pts=sim_parts_per_pts.detach().numpy()
    img_parts_per_pts=np.zeros((sim_parts_per_pts.shape[0],sim_parts_per_pts.shape[1]))
    for h in range(sim_parts_per_pts.shape[0]):
        for w in range(sim_parts_per_pts.shape[1]):
            img_parts_per_pts[h][w]= sim_parts_per_pts[h][w]
    
    block_size = 1
    if block_size !=1:
        image_blocked_l = np.kron(img_l, np.ones((block_size, block_size)))
        image_l = image_blocked_l
        image_blocked_r = np.kron(img_r, np.ones((block_size, block_size)))
        image_r = image_blocked_r
        img_parts_per_pts = np.kron(img_parts_per_pts, np.ones((block_size, block_size)))

    else:
        image_l, image_r, image_parts_per_pts = img_l, img_r, img_parts_per_pts
        
    return image_l, image_r, image_parts_per_pts

count=0
ziku_label=np.array([])
filename_befor=""
for filename in np.array(label_base):
    
    if filename[:2] != filename_befor:
        ziku_label=np.append(ziku_label,count)
        filename_befor=filename[:2]
    count+=1




ziku_label=ziku_label
ziku_label[0]=0
ziku_label=np.append(ziku_label,len(label_base))
print(ziku_label)

import matplotlib.colors as mcolors
#image=parts2ges_all(pts_base,hands_base,seg_base)
cate_color=["white","red","pink","orange",""]
il,ir,i_parts2pts=parts2ges(pts_base,hands_base,seg_base)
plt.rcParams["font.size"]=15

plt.figure(1)
plt.figure(figsize=(8, 8)) 
plt.imshow(il, cmap='viridis', aspect='equal',vmax=1,vmin=0)
labels =[int(round(tick)) for tick in ziku_label] 
plt.colorbar()  # カラーバーを表示
plt.xticks(ticks=ziku_label-0.5,labels=labels)
plt.xticks(rotation=90)
plt.yticks(ticks=ziku_label-0.5,labels=labels)
plt.savefig('colormap_image.png', dpi=300)
"""
plt.xlabel("gesture_feat")
plt.ylabel("parts_feat")
"""

plt.tick_params(labelsize=8.5)
# グラフを表示



plt.figure(2)
plt.figure(figsize=(8, 8)) 
plt.imshow(ir, cmap='viridis', aspect='equal',vmax=1,vmin=0)
labels =[int(round(tick)) for tick in ziku_label] 
plt.colorbar()  # カラーバーを表示
plt.xticks(ticks=ziku_label-0.5,labels=labels)
plt.xticks(rotation=90)
plt.yticks(ticks=ziku_label-0.5,labels=labels)
"""
plt.xlabel("gesture_feat")
plt.ylabel("parts_feat")
"""
plt.tick_params(labelsize=8.5)


plt.figure(3)
plt.figure(figsize=(8, 8)) 
plt.imshow(i_parts2pts, cmap='viridis', aspect='equal',vmax=1,vmin=0)
labels =[int(round(tick)) for tick in ziku_label] 
plt.colorbar()  # カラーバーを表示
plt.xticks(ticks=ziku_label-0.5,labels=labels)
plt.xticks(rotation=90)
plt.yticks(ticks=ziku_label-0.5,labels=labels)
"""
plt.xlabel("gesture_feat")
plt.ylabel("parts_feat")
"""
plt.tick_params(labelsize=8.5)
plt.show()



