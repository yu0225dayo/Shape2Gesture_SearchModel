"""
Training script for Parts2Gesture contrastive learning models.

Trains PointNet-based models for gesture recognition from 3D point cloud parts.
"""

from __future__ import print_function
import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import ShapeNetDataset
from model import PointNetDenseCls, feature_transform_regularizer, ContrastiveNet, PartsToPtsNet
from functions import load_models

def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, top_k: Tuple = (1, 5)) -> Dict:
    """
    Calculate top-k accuracy for contrastive learning.
    
    Args:
        logits: Similarity score matrix (N x N)
        labels: Ground truth labels (torch.arange(N))
        top_k: Tuple of k values for top-k accuracy
    
    Returns:
        Dictionary of accuracy scores
    """
    max_k = max(top_k)
    batch_size = logits.size(0)

    # Get top-k indices
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)

    # Check if predictions match ground truth
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))

    # Calculate accuracy for each k
    accuracies = {}
    for k in top_k:
        correct_k = correct[:, :k].reshape(-1).float().sum(0, keepdim=True)
        accuracies[f'Top-{k}'] = correct_k.mul_(100.0 / batch_size).item()

    return accuracies


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='Contratstive_Parts2Gesture', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default="dataset", help="dataset path")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        data_augmentation=True)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    

    Writer = SummaryWriter(log_dir="./save_log")

    
    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        split='val',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = dataset.num_seg_classes
    print('classes', num_classes)

    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    # モデル初期化
    if opt.model != '' and os.path.exists(opt.model):
        # 既存モデルを読み込む場合
        pointnet_classifier, sk_parts_classifier, parts2pts_classifier = load_models(opt.model)
    else:
        # 新規モデルを初期化
        pointnet_classifier = PointNetDenseCls(k=3, feature_transform=opt.feature_transform)
        sk_parts_classifier = ContrastiveNet()
        parts2pts_classifier = PartsToPtsNet()

    optimizer = optim.Adam([{"params":pointnet_classifier.parameters()},{"params":sk_parts_classifier.parameters()},{"params":parts2pts_classifier.parameters()}] ,  lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    pointnet_classifier.cuda()
    sk_parts_classifier.cuda()
    parts2pts_classifier.cuda()

    num_batch = len(dataset) / opt.batchSize
    best_partseg_acc = 0
    best_ps_acc = 0
    best_p2pts_acc = 0
    
    min_loss_partseg = 3
    min_loss_contrastive = 5
    min_loss_total = 11
    min_loss_p2pts = 1

    for epoch in range(opt.nepoch):
        #scheduler.step()
        for i, data in enumerate(dataloader, 0):
            #print(i, data)
            points, target, hands, filename ,loss_weight = data
            #全体のparts:0の重要度(重み)を減らしたい。
            loss_weight = 1 / (torch.sum(loss_weight,dim=0) / opt.batchSize)
            
            #print(points.shape, target.shape,hands.shape)
            points = points.transpose(2, 1)
            partsseg_target=target
            
            points, target, loss_weight = points.cuda(), target.cuda() ,loss_weight.cuda()
            optimizer.zero_grad()
            pointnet_classifier = pointnet_classifier.train()
            #parts seg
            pred, trans, trans_feat,all_feat = pointnet_classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            
            loss_pointnet =F.nll_loss(pred, target , weight = loss_weight)
            if opt.feature_transform:
                loss_pointnet += feature_transform_regularizer(trans_feat) * 0.001

            #parts---hand
            
            pred_choice = pred.data.max(1)[1]
            pred_np = pred_choice.cpu().data.numpy()
            # pred_choice 1 :left,  pred_choice 2 : right
            
            points=points.transpose(1, 2).cpu().data.numpy()
            pred_np=pred_np.reshape(opt.batchSize,2048,1)
            
            pl=np.array([])
            pr=np.array([])
            
            for batch in range(opt.batchSize):
                count=0
                parts_l_list=np.array([])
                parts_r_list=np.array([])

                #print(batch,np.count_nonzero(pred_np[batch]==1),np.count_nonzero(pred_np[batch]==2) )
                #右、左手のラベル1,2が明らかに推測できない時に教師をそのまま用いる。
                if np.count_nonzero(pred_np[batch]==2)<=10:    
                    target_l=partsseg_target
                else:
                    target_l=pred_np
                if np.count_nonzero(pred_np[batch]==1)<=10:
                    target_r=partsseg_target
                else:
                    target_r=pred_np
                
                for j in range(2048):
                    if target_l[batch][j]==2:
                        parts_l_list=np.append(parts_l_list,points[batch][j])
                    if target_r[batch][j]==1:
                        parts_r_list=np.append(parts_r_list,points[batch][j])

                while len(parts_l_list)<=(3 * 256):
                    add_list=parts_l_list*1.01
                    parts_l_list=np.append(parts_l_list,add_list)
                while len(parts_r_list)<=(3 * 256):
                    add_list=parts_r_list*1.01
                    parts_r_list=np.append(parts_r_list,add_list)
                
                #sampling
                parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3) 
                parts_r_list=parts_r_list.reshape(int(len(parts_r_list)/3),3)              
                choice_l = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True)                
                choice_r = np.random.choice(int(parts_r_list.shape[0]), 256, replace=True)   
                pl=np.append(pl, parts_l_list[choice_l,:])             
                pr=np.append(pr,parts_r_list[choice_r,:])
                pl = pl - np.expand_dims(np.mean(pl, axis = 0), 0)
                pr = pr - np.expand_dims(np.mean(pr, axis = 0), 0)

            pl=pl.reshape(opt.batchSize,256,3).astype(np.float32)
            pr=pr.reshape(opt.batchSize,256,3).astype(np.float32)
            pl=torch.from_numpy(pl)
            pr=torch.from_numpy(pr)
            pl=pl.transpose(2,1)
            pr=pr.transpose(2,1)

            #ジェスチャの前処理
            hand=np.split(hands,2,axis=1)
            hand_l=hand[0]
            hand_r=hand[1]
            #手首を0に
            for k in range(opt.batchSize):
                hand_l[k] = hand_l[k] - hand_l[k][0]
                hand_r[k] = hand_r[k] - hand_r[k][0]
        
            hand_l=hand_l.reshape(opt.batchSize,69)
            hand_r=hand_r.reshape(opt.batchSize,69)

            hand_l, hand_r, parts_l, parts_r, all_feat = hand_l.cuda(), hand_r.cuda(), pl.cuda(), pr.cuda(), all_feat.cuda()
            #train parts2ges
            #input : input_sk,input_parts,input_all_feat   out: logit_per_sk, logit_per_parts
            sk_parts_classifier = sk_parts_classifier.train()
            logit_per_sk_l, logit_per_parts_l, sk_feat_l, parts_feat_l = sk_parts_classifier(hand_l, parts_l, all_feat)
            logit_per_sk_r, logit_per_parts_r, sk_feat_r, parts_feat_r = sk_parts_classifier(hand_r, parts_r, all_feat)
            
            ans = torch.eye(opt.batchSize,opt.batchSize).cuda()
            loss_sk_l = F.cross_entropy(logit_per_sk_l,ans)
            loss_parts_l = F.cross_entropy(logit_per_parts_l,ans)
            
            loss_sk_r = F.cross_entropy(logit_per_sk_r,ans)
            loss_parts_r = F.cross_entropy(logit_per_parts_r,ans)

            #parts2pts
            parts2pts_classifier= parts2pts_classifier.train()
            logit_per_p, logit_per_pts = parts2pts_classifier(parts_feat_l, parts_feat_r, all_feat)
            loss_p2pts = (F.cross_entropy(logit_per_p, ans) + F.cross_entropy(logit_per_pts,ans)) / 2

            loss_sk_parts = (loss_sk_l + loss_parts_l + loss_sk_r + loss_parts_r)/4
            loss = loss_sk_parts + loss_pointnet + loss_p2pts
            
            loss.backward()
            optimizer.step()

            #parts segmentation acc
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print("p",points.shape,"t",target.shape,"pred",pred_choice,"trans",trans.shape,"tansfeat",trans_feat)

            partseg_acc=correct.item()/float(opt.batchSize * 2048)*100
            loss_partseg=loss_pointnet.item()

            print('[%d: %d/%d] total-train loss: %f' % (epoch, i, num_batch, loss.item()))
            print('[%d: %d/%d] partsseg loss: %f accuracy: %f' % (epoch, i, num_batch, loss_partseg, partseg_acc)) 

            #parts2ges acc
            #logitの各行の最大の値がansの1の場所にある？
            ps_l=(torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_sk_l,dim=1)).cpu().sum()) / (opt.batchSize)
            pl_l=(torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_parts_l,dim=1)).cpu().sum()) / (opt.batchSize)
            ps_r=(torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_sk_r,dim=1)).cpu().sum()) / (opt.batchSize)
            pl_r=(torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_parts_r,dim=1)).cpu().sum()) / (opt.batchSize)

            ps_acc = (ps_l + pl_l + ps_r + pl_r)/4
            ps_loss = loss_sk_parts.item()

            class_same_count=0
            for j in range(opt.batchSize):
                if filename[j][:2] == filename[int(torch.argmax(logit_per_sk_l,dim=1).cpu()[j])][:2]:
                    class_same_count+=1
                if filename[j][:2] == filename[int(torch.argmax(logit_per_parts_l,dim=1).cpu()[j])][:2]:
                    class_same_count+=1
                if filename[j][:2] == filename[int(torch.argmax(logit_per_sk_r,dim=1).cpu()[j])][:2]:
                    class_same_count+=1
                if filename[j][:2] == filename[int(torch.argmax(logit_per_parts_l,dim=1).cpu()[j])][:2]:
                    class_same_count+=1

            class_same_acc_g2p= class_same_count / opt.batchSize / 4 * 100
            print('[%d: %d/%d] parts-sk loss: %f accuracy: %f class_same_acc %f ' % (epoch, i, num_batch, ps_loss, ps_acc*100,class_same_acc_g2p)) 
            
            #parts2shape acc(logit)
            loss_p2pts=loss_p2pts.item()
            lp = (torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_p,dim=1)).cpu().sum()) / (opt.batchSize)
            lpts = (torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_pts,dim=1)).cpu().sum()) / (opt.batchSize)
            p2pts_acc=(lp + lpts) /2

            #accuracy (class)
            class_same_count=0
            for j in range(opt.batchSize):
                if filename[j][:2] == filename[int(torch.argmax(logit_per_p,dim=1).cpu()[j])][:2]:
                    class_same_count+=1
                if filename[j][:2] == filename[int(torch.argmax(logit_per_pts,dim=1).cpu()[j])][:2]:
                    class_same_count+=1


            class_same_acc_p2p= class_same_count / opt.batchSize / 2 * 100
            print('[%d: %d/%d] parts2pts loss: %f accuracy: %f class_same_acc %f ' % (epoch, i, num_batch, loss_p2pts, p2pts_acc*100,class_same_acc_p2p)) 

            # loss
            Writer.add_scalars("tensorboad/loss_total",{"train":loss},epoch)
            Writer.add_scalars("tensorboad/loss_partsseg",{"train":loss_partseg},epoch)
            Writer.add_scalars("tensorboad/loss_sk_pts",{"train":ps_loss},epoch)
            Writer.add_scalars("tensorboad/loss_p2pts",{"train":loss_p2pts},epoch)
            # acc
            Writer.add_scalars("tensorboad/acc_partseg",{"train":partseg_acc},epoch)
            Writer.add_scalars("tensorboad/acc_sk_parts",{"train":ps_acc*100},epoch)
            Writer.add_scalars("tensorboad/class_same_acc_ges2parts",{"train":class_same_acc_g2p},epoch)
            Writer.add_scalars("tensorboad/acc_p2pts",{"train":p2pts_acc},epoch)
            Writer.add_scalars("tensorboad/class_same_acc_parts2pts",{"train":class_same_acc_p2p},epoch)

            if (i+1) % 9 == 0:
                j, data = next(enumerate(testdataloader,0))
                
                points, target, hands, label, loss_weight = data
                loss_weight = 1 / torch.sum(loss_weight,dim=0)/opt.batchSize
                points = points.transpose(2, 1)
                partsseg_target=target
                
                points, target , loss_weight = points.cuda(), target.cuda(), loss_weight.cuda()
                pointnet_classifier = pointnet_classifier.eval()
                
                #parts seg
                pred, trans, trans_feat,all_feat = pointnet_classifier(points)
                
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0]
                
                loss_pointnet =F.nll_loss(pred, target , weight = loss_weight)
                if opt.feature_transform:
                    loss_pointnet += feature_transform_regularizer(trans_feat) * 0.001
            
                #parts---hand
                
                pred_choice = pred.data.max(1)[1]
                pred_np = pred_choice.cpu().data.numpy()
                # pred_choice 1 :left,  pred_choice 2 : right
                points=points.transpose(1, 2).cpu().data.numpy()
                pred_np=pred_np.reshape(opt.batchSize,2048,1)
                
                pl=np.array([])
                pr=np.array([])
                for batch in range(opt.batchSize):
                    
                    parts_l_list=np.array([])
                    parts_r_list=np.array([])
                    if np.count_nonzero(pred_np[batch]==2) <=10:    
                        target_l=partsseg_target
                    else:
                        target_l=pred_np
                    if np.count_nonzero(pred_np[batch]==1) <=10:
                        target_r=partsseg_target
                    else:
                        target_r=pred_np
                    
                    for j in range(2048):
                        if target_l[batch][j]==2:
                            parts_l_list=np.append(parts_l_list,points[batch][j])
                        if target_r[batch][j]==1:
                            parts_r_list=np.append(parts_r_list,points[batch][j])
                    
                    while len(parts_l_list)<=(3 * 256):
                        add_list=parts_l_list*1.01
                        parts_l_list=np.append(parts_l_list,add_list)
                    while len(parts_r_list)<=(3 * 256):
                        count+=1
                        add_list=parts_r_list*1.01
                        parts_r_list=np.append(parts_r_list,add_list)
                        
                    #sampling
                    parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3) 
                    parts_r_list=parts_r_list.reshape(int(len(parts_r_list)/3),3)              
                    choice_l = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True)                
                    choice_r = np.random.choice(int(parts_r_list.shape[0]), 256, replace=True)   
                    pl=np.append(pl, parts_l_list[choice_l,:])             
                    pr=np.append(pr,parts_r_list[choice_r,:])
                    
                    pl = pl - np.expand_dims(np.mean(pl, axis = 0), 0)
                    pr = pr - np.expand_dims(np.mean(pr, axis = 0), 0)

                pl=pl.reshape(opt.batchSize,256,3).astype(np.float32)
                pr=pr.reshape(opt.batchSize,256,3).astype(np.float32)

                pl=torch.from_numpy(pl)
                pr=torch.from_numpy(pr)
                pl=pl.transpose(2,1)
                pr=pr.transpose(2,1)

                hand=np.split(hands,2,axis=1)
                hand_l = hand[0]
                hand_r = hand[1]

                for k in range(opt.batchSize):
                    hand_l[k] = hand_l[k] - hand_l[k][0]
                    hand_r[k] = hand_r[k] - hand_r[k][0]

                hand_l=hand_l.reshape(opt.batchSize,69)
                hand_r=hand_r.reshape(opt.batchSize,69)


                #print(hand_l.shape,hand_r.shape, type(hand_l))
                hand_l, hand_r, parts_l, parts_r, all_feat = hand_l.cuda(), hand_r.cuda(), pl.cuda(), pr.cuda(), all_feat.cuda()
                sk_parts_classifier = sk_parts_classifier.eval()
                logit_per_sk_l, logit_per_parts_l, sk_feat_l, parts_feat_l = sk_parts_classifier(hand_l, parts_l, all_feat)
                logit_per_sk_r, logit_per_parts_r, sk_feat_r, parts_feat_r = sk_parts_classifier(hand_r, parts_r, all_feat)

                ans = torch.eye(opt.batchSize,opt.batchSize).cuda()
                loss_sk_l = F.cross_entropy(logit_per_sk_l,ans)
                loss_parts_l = F.cross_entropy(logit_per_parts_l,ans)
                loss_sk_r = F.cross_entropy(logit_per_sk_r,ans)
                loss_parts_r = F.cross_entropy(logit_per_parts_r,ans)
                
                #parts2pts
                parts2pts_classifier = parts2pts_classifier.eval()
                logit_per_p, logit_per_pts = parts2pts_classifier(parts_feat_l, parts_feat_r, all_feat)
                loss_p2pts = (F.cross_entropy(logit_per_p, ans) + F.cross_entropy(logit_per_pts,ans)) / 2
                loss_sk_parts = (loss_sk_l + loss_parts_l + loss_sk_r + loss_parts_r)/4
                loss = loss_sk_parts + loss_pointnet + loss_p2pts
            
                #parts segmentation acc
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()

                print("p",points.shape,"t",target.shape,"pred",pred_choice,"trans",trans.shape,"tansfeat",trans_feat)

                val_acc=correct.item()/float(opt.batchSize * 2048)*100
                loss_partseg=loss_pointnet.item()
                print('[%d: %d/%d] %s total-test loss: %f' % (epoch, i, num_batch, blue("test"), loss.item()))
                print('[%d: %d/%d] %s partssesg loss: %f accuracy: %f' % (epoch, i, num_batch, blue("test"),loss_partseg, val_acc)) 

    
                #parts --- sk acc
                #logitの各行の最大の値がlabelの1の場所にある？
                ps_l=(torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_sk_l,dim=1)).cpu().sum()) / (opt.batchSize)
                pl_l=(torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_parts_l,dim=1)).cpu().sum()) / (opt.batchSize)
                ps_r=(torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_sk_r,dim=1)).cpu().sum()) / (opt.batchSize)
                pl_r=(torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_parts_r,dim=1)).cpu().sum()) / (opt.batchSize)

                ps_acc=(ps_l + pl_l + ps_r + pl_r)/4 *100
                ps_loss=loss_sk_parts.item()
                class_same_count=0

                for j in range(opt.batchSize):
                    if filename[j][:2] == filename[int(torch.argmax(logit_per_sk_l,dim=1).cpu()[j])][:2]:
                        class_same_count+=1
                    if filename[j][:2] == filename[int(torch.argmax(logit_per_parts_l,dim=1).cpu()[j])][:2]:
                        class_same_count+=1
                    if filename[j][:2] == filename[int(torch.argmax(logit_per_sk_r,dim=1).cpu()[j])][:2]:
                        class_same_count+=1
                    if filename[j][:2] == filename[int(torch.argmax(logit_per_parts_l,dim=1).cpu()[j])][:2]:
                        class_same_count+=1

                class_same_acc=class_same_count / opt.batchSize / 4 *100
                print('[%d: %d/%d] %s parts-sk loss: %f accuracy: %f class_same_acc %f ' % (epoch, i, num_batch,blue("test"), ps_loss, ps_acc,class_same_acc)) 

                #parts2pts
                p2pts_loss=loss_p2pts.item()

                lp = (torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_p,dim=1)).cpu().sum()) / (opt.batchSize)
                lpts = (torch.argmax(ans,dim=1).eq(torch.argmax(logit_per_pts,dim=1)).cpu().sum()) / (opt.batchSize)
                p2pts_acc=(lp + lpts) /2

                class_same_count=0
                for j in range(opt.batchSize):
                    if filename[j][:2] == filename[int(torch.argmax(logit_per_p,dim=1).cpu()[j])][:2]:
                        class_same_count+=1
                    if filename[j][:2] == filename[int(torch.argmax(logit_per_pts,dim=1).cpu()[j])][:2]:
                        class_same_count+=1

                class_same_acc_p2p= class_same_count / opt.batchSize / 2 * 100
                print('[%d: %d/%d] %s parts2pts loss: %f accuracy: %f class_same_acc %f ' % (epoch, i, num_batch, blue("test"),p2pts_loss, p2pts_acc*100,class_same_acc_p2p)) 

                # eval スコアに基づくモデル保存
                if val_acc > best_partseg_acc:
                    print("-----------")
                    print("best_partseg_accを更新（eval: {:.2f}%）".format(val_acc))
                    torch.save(pointnet_classifier.state_dict(), '%s/pointnet_model_acc_partseg_best.pth' % (opt.outf))
                    torch.save(sk_parts_classifier.state_dict(), '%s/contrastive_model_acc_partseg_best.pth' % (opt.outf))
                    torch.save(parts2pts_classifier.state_dict(), '%s/parts2pts_model_acc_partseg_best.pth' % (opt.outf))
                    best_partseg_acc = val_acc

                if loss_partseg < min_loss_partseg:
                    print("----------")
                    print("min_loss_partsegを更新（eval）")
                    torch.save(pointnet_classifier.state_dict(), '%s/pointnet_model_loss_partseg_best.pth' % (opt.outf))
                    torch.save(sk_parts_classifier.state_dict(), '%s/contrastive_model_loss_partseg_best.pth' % (opt.outf))
                    torch.save(parts2pts_classifier.state_dict(), '%s/parts2pts_model_loss_partseg_best.pth' % (opt.outf))
                    min_loss_partseg = loss_partseg

                # total loss
                if loss < min_loss_total:
                    print("----------")
                    print("min_loss_totalを更新（eval）")
                    torch.save(pointnet_classifier.state_dict(), '%s/pointnet_model_loss_total_best.pth' % (opt.outf))
                    torch.save(sk_parts_classifier.state_dict(), '%s/contrastive_model_loss_total_best.pth' % (opt.outf))
                    torch.save(parts2pts_classifier.state_dict(), '%s/parts2pts_model_loss_total_best.pth' % (opt.outf))
                    min_loss_total = loss

                # parts-sk accuracy
                if ps_acc > best_ps_acc:
                    print("----------")
                    print("best_ps_accを更新（eval: {:.2f}%）".format(ps_acc))
                    torch.save(pointnet_classifier.state_dict(), '%s/pointnet_model_loss_contrastive_best.pth' % (opt.outf))
                    torch.save(sk_parts_classifier.state_dict(), '%s/contrastive_model_loss_contrastive_best.pth' % (opt.outf))
                    torch.save(parts2pts_classifier.state_dict(), '%s/parts2pts_model_loss_contrastive_best.pth' % (opt.outf))
                    best_ps_acc = ps_acc
                
                # parts2pts accuracy
                if p2pts_acc > best_p2pts_acc:
                    print("----------")
                    print("best_p2pts_accを更新（eval: {:.2f}%）".format(p2pts_acc*100))
                    torch.save(pointnet_classifier.state_dict(), '%s/pointnet_model_loss_parts2pts_best.pth' % (opt.outf))
                    torch.save(sk_parts_classifier.state_dict(), '%s/contrastive_model_loss_parts2pts_best.pth' % (opt.outf))
                    torch.save(parts2pts_classifier.state_dict(), '%s/parts2pts_model_loss_parts2pts_best.pth' % (opt.outf))
                    best_p2pts_acc = p2pts_acc
                
                # loss
                Writer.add_scalars("tensorboad/loss_total",{"val":loss},epoch)
                Writer.add_scalars("tensorboad/loss_partsseg",{"val":loss_partseg},epoch)
                Writer.add_scalars("tensorboad/loss_sk_pts",{"val":ps_loss},epoch)
                Writer.add_scalars("tensorboad/loss_p2pts",{"val":p2pts_loss},epoch)

                # acc
                Writer.add_scalars("tensorboad/acc_partseg",{"val":val_acc},epoch)
                Writer.add_scalars("tensorboad/acc_sk_parts",{"val":ps_acc},epoch)
                Writer.add_scalars("tensorboad/class_same_acc",{"val":class_same_acc},epoch)
                Writer.add_scalars("tensorboad/acc_p2pts",{"val":p2pts_acc},epoch)
                Writer.add_scalars("tensorboad/class_same_acc_parts2pts",{"val":class_same_acc_p2p},epoch)
   
        scheduler.step()
    Writer.close()

    print("----------")
    print("学習終了")
    torch.save(pointnet_classifier.state_dict(), '%s/pointnet_model_final.pth' % (opt.outf))
    torch.save(sk_parts_classifier.state_dict(), '%s/contrastive_model_final.pth' % (opt.outf))
    torch.save(parts2pts_classifier.state_dict(), '%s/parts2pts_model_final.pth' % (opt.outf))
