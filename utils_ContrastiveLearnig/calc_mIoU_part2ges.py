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

parser.add_argument('--model', type=str, default='', help='model path')
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

#すべてのジェスチャーの中のどれと一番マッチするか
def parts2ges_all(p,g,seg):
    outarr_p_size=p.shape[0]
    outarr_g_size=g.shape[0]
    points =p.transpose(2, 1).contiguous()
    pred, _, _,all_feat = pointnet_classifier(points)
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
        #augmentation 検出パーツが少ないとき、性能が変わりそう
        # *1.01は少しだけずらす意味
        
        while len(parts_l_list)<=(3 * 256):
            #print("augment")
            count+=1
            add_list=parts_l_list*1.01
            parts_l_list=np.append(parts_l_list,add_list)
        parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3)             
        choice = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True)                
        pl=np.append(pl,parts_l_list[choice,:])
        while len(parts_r_list)<=(3 * 256):
            #print("augment")
            count+=1
            add_list=parts_r_list*1.01
            parts_r_list=np.append(parts_r_list,add_list)
        
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



def parts2ges(p,g,seg,filename):
    batchsize=p.shape[0]
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
        #augmentation 検出パーツが少ないとき、性能が変わりそう
        # *1.01は少しだけずらす意味
        
        while len(parts_l_list)<=(3 * 256):
            #print("augment")
            count+=1
            add_list=parts_l_list*1.01
            parts_l_list=np.append(parts_l_list,add_list)
        parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3)             
        choice = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True)                
        pl=np.append(pl,parts_l_list[choice,:])
        while len(parts_r_list)<=(3 * 256):
            #print("augment")
            count+=1
            add_list=parts_r_list*1.01
            parts_r_list=np.append(parts_r_list,add_list)
        
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

    ans = torch.eye(batchsize,batchsize)
    pl_gl=(torch.argmax(ans,dim=1).eq(torch.argmax(sim_pl_gl,dim=1)).sum()) / (batchsize)
    pr_gr=(torch.argmax(ans,dim=1).eq(torch.argmax(sim_pr_gr,dim=1)).sum()) / (batchsize)

    sim_plgl=torch.argmax(sim_pl_gl,dim=1)
    sim_prgr=torch.argmax(sim_pr_gr,dim=1)
    same_arr_r=np.array([])
    same_arr_l=np.array([])


    for i in range(batchsize):
        if sim_plgl[i]==i:
            same_arr_l=np.append(same_arr_l,1)
        if sim_prgr[i]==i:
            same_arr_r=np.append(same_arr_r,1)
        if sim_plgl[i]!=i:
            same_arr_l=np.append(same_arr_l,0)
        if sim_prgr[i]!=i:
            same_arr_r=np.append(same_arr_r,0)
    

    same_arr_class_l=np.array([])
    same_arr_class_r=np.array([])

    for j in range(batchsize):
        
        if filename[j][:2] == filename[int(torch.argmax(sim_pl_gl,dim=1)[j])][:2]:
            same_arr_class_l=np.append(same_arr_class_l,1)
        if filename[j][:2] == filename[int(torch.argmax(sim_pr_gr,dim=1)[j])][:2]:
            same_arr_class_r=np.append(same_arr_class_r,1)
        if filename[j][:2] != filename[int(torch.argmax(sim_pl_gl,dim=1)[j])][:2]:
            same_arr_class_l=np.append(same_arr_class_l,0)
        if filename[j][:2] != filename[int(torch.argmax(sim_pr_gr,dim=1)[j])][:2]:
            same_arr_class_r=np.append(same_arr_class_r,0)
    
    #parts2pts
    sim_parts_per_pts, _ = p2pts_classifier(parts_l_feat, parts_r_feat, all_feat)
    same_p2p =np.array([])  
    sim_p2p=torch.argmax(sim_parts_per_pts,dim=1)

    for k in range(batchsize):
        if sim_p2p[k]==k:
            same_p2p = np.append(same_p2p,1)
        if sim_p2p[k]!=k:
            same_p2p = np.append(same_p2p,0)
    same_class_p2p = np.array([])

    for l in range(batchsize):
        if filename[l][:2] == filename[int(torch.argmax(sim_parts_per_pts,dim=1)[l])][:2]:
            same_class_p2p=np.append(same_class_p2p,1)
        if filename[l][:2] != filename[int(torch.argmax(sim_parts_per_pts,dim=1)[l])][:2]:
            same_class_p2p=np.append(same_class_p2p,0)

    return same_arr_l, same_arr_r ,same_arr_class_l, same_arr_class_r, same_p2p, same_class_p2p
arr_name=np.array([])
for csv in label_base:
    arr_name = np.append(arr_name,csv[:2])
arr_l, arr_r, s_arr_l, s_arr_r , p2p, s_p2p=parts2ges(pts_base,hands_base,seg_base, label_base)

arr_out_l=np.stack((arr_l,arr_name),1)
arr_out_r=np.stack((arr_r,arr_name),1)

sarr_out_l=np.stack((s_arr_l,arr_name),1)
sarr_out_r=np.stack((s_arr_r,arr_name),1)

arr_out_p2p=np.stack((p2p,arr_name),1)
sarr_out_p2p=np.stack((s_p2p,arr_name),1)

def calc_mIoU(arr_out,mode):
    classes = np.unique(arr_out[:, 1])
    # 各クラスの平均を計算
    averages = {}
    for cls in classes:
        # クラスに対応する行を取得
        values = arr_out[arr_out[:, 1] == cls, 0].astype(np.float32)
        
        # 平均を計算
        averages[cls] = {"mean": np.mean(values),
                         "num":len(values)}
        #　クラスの要素数を取得
        


    # 結果を表示
    for cls, avg in averages.items():
        print(f"{mode} {cls} の平均: {avg['mean']}, 要素数: {avg['num']}")

        
print("------------")
calc_mIoU(arr_out_l,"same_plgl")
print("------------")
calc_mIoU(arr_out_r,"same_prgr")
print("------------")
calc_mIoU(sarr_out_l,"same_class_plgl")
print("------------")
calc_mIoU(sarr_out_r,"same_class_prgr")
print("------------")
calc_mIoU(arr_out_p2p,"same_p2p")
print("------------")
calc_mIoU(sarr_out_p2p,"same_class_p2p")



