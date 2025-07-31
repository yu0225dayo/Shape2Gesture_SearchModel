from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import pandas as pd


def get_segmentation_classes(root,mode):
    #mode=train or test
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    #root=data_directory
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            
            cat[ls[0]] = ls[0]
            #cat=ラベル番号

    for item in cat:
        dir_mode=os.path.join(root,cat[item],mode)
        dir_seg = os.path.join(dir_mode, 'points_label')
        dir_point = os.path.join(dir_mode,'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.off'), os.path.join(dir_seg, token + '.seg')))#.pts,.segファイルの拡張子はなでもok

    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ShapeNetDataset2(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split=split

        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        self.name=class_choice[0]
        datafol=os.path.join(self.root,split)
        print(datafol)
        
        self.classes = {self.name:20}
        print(self.classes)
        self.datapath=[]

        namefol=os.path.join(datafol,"pts")
        
        self.cate = []
        self.cate_num =[]
        if self.split == "train":
            for csv in os.listdir(namefol):
                if self.cate == [] or self.cate[-1][:2] != csv[:2]:
                    self.cate.append(csv[:2])
                    self.cate_num.append(1)
                else:
                    self.cate_num[-1] +=1
            print(self.cate , self.cate_num, max(self.cate_num))
            #リピート回数
            self.repeat = [(max(self.cate_num) // cate_num) for cate_num in self.cate_num]
            print(self.repeat)
            dict_fol = dict(zip(self.cate , self.repeat))
            #データ数を合わせる
            for csv in os.listdir(namefol):
                print(csv[:2])
                k = dict_fol[csv[:2]]
                for j in range(k):
                    pts=os.path.join(datafol,"pts",csv)
                    csvname , ext = os.path.splitext(csv)
                    label=os.path.join(datafol,"label",csvname+"_label"+ext)
                    hand=os.path.join(datafol,"hands",csvname+"_hand"+ext)
                    self.datapath.append([self.name,pts,label,hand,csvname])
                    
        else:
            for csv in os.listdir(namefol):
                pts=os.path.join(datafol,"pts",csv)
                csvname , ext = os.path.splitext(csv)
                label=os.path.join(datafol,"label",csvname+"_label"+ext)
                hand=os.path.join(datafol,"hands",csvname+"_hand"+ext)
                self.datapath.append([self.name,pts,label,hand,csvname])

        self.seg_classes, self.num_seg_classes=self.name,3
        print("self.seg_classes, self.num_seg_classes:",self.seg_classes, self.num_seg_classes)
        

    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set=np.array(pd.read_csv(fn[1],header=None)).astype(np.float32)
        cls = self.classes[self.datapath[index][0]]
        #handを左右で分ける
        
        hand_set = np.array(pd.read_csv(fn[3],header=None)).astype(np.float32)
        label=fn[4]

        hand=np.split(hand_set,2,axis=0)
        #--------変更-------------

        
        hand_l = hand[0]
        hand_r = hand[1]

        middle_len =sum(np.array([np.linalg.norm(hand_l[0]-hand_l[8]),
                     np.linalg.norm(hand_l[8]-hand_l[9]),
                     np.linalg.norm(hand_l[9]-hand_l[10]),
                     np.linalg.norm(hand_l[10]-hand_l[20])]))
        
        hand_scale = 1 / middle_len 
        hand_l =hand_l * hand_scale / 2

        middle_len =sum(np.array([np.linalg.norm(hand_r[0]-hand_r[8]),
                     np.linalg.norm(hand_r[8]-hand_r[9]),
                     np.linalg.norm(hand_r[9]-hand_r[10]),
                     np.linalg.norm(hand_r[10]-hand_r[20])]))
        
        hscale_r = 1 /middle_len 

        hand_r = hand_r * hscale_r /2
        
        hand_set = np.vstack((hand_l, hand_r))

        try:
            seg=pd.read_csv(fn[2],header=None).to_numpy().astype(np.int64)
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            #resample
            point_set = point_set[choice, :]
            

            point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
            point_set = point_set / dist #scale
            hand_set_rote = hand_set

            if self.data_augmentation:
                #print("augment")
                theta = np.random.uniform(0,np.pi*2)
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
                #print(rotation_matrix)
                random_jitter=np.random.normal(0, 0.02, size=point_set.shape)
                point_set[:,[0,1]] = point_set[:,[0,1]].dot(rotation_matrix) # random rotation
                point_set += random_jitter # random jitter
                
                hand_set_rote[:,[0,1]] = hand_set_rote[:,[0,1]].dot(rotation_matrix)
                #hand_set += np.random.normal(0, 0.02, size=hand_set.shape)

            seg = seg[choice]
            batch_weight=np.array([np.count_nonzero(seg==0), np.count_nonzero(seg==1),np.count_nonzero(seg==2)] ) / len(seg)
            point_set = torch.from_numpy(point_set.astype(np.float32))
            seg = torch.from_numpy(seg)
            cls = torch.from_numpy(np.array([cls]).astype(np.int64))
            hand_set_rote=torch.from_numpy(hand_set_rote.astype(np.float32))
            batch_weight = torch.from_numpy(batch_weight.astype(np.float32))
            
            if self.classification:
                return point_set, cls
            else:
                return point_set, seg, hand_set_rote,label, batch_weight, hand_set,hand_scale

        except Exception as e:
            print(f"[Error index:{index}]:", e)
            #---------新しい入力に対して---------
            print("推測されたものを出力")
            fn = self.datapath[index]
            point_set=np.array(pd.read_csv(fn[1],header=None)).astype(np.float32)
            cls = self.classes[self.datapath[index][0]]

            #print(point_set[1])

            #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
            #resample
            #point_set = point_set[choice, :]

            point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
            
            point_set = point_set / dist #scale

            if self.data_augmentation:
                theta = np.random.uniform(0,np.pi*2)
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
                point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
                point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

            #seg = seg[choice]
            point_set = torch.from_numpy(point_set)
            #seg = torch.from_numpy(seg)
            #cls = torch.from_numpy(np.array([cls]).astype(np.int64))

            if self.classification:
                return point_set
            else:
                return point_set

     
        #print(point_set[1])
        
  

    def __len__(self):
        return len(self.datapath)

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split=split

        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        self.name=class_choice
        datafol=os.path.join(self.root,split)
        print(datafol)
        
        self.classes = {self.name:20}
        print(self.classes)
        self.datapath=[]

        namefol=os.path.join(datafol,"pts")
        
        #print(namefol)
        for csv in os.listdir(namefol):
            pts=os.path.join(datafol,"pts",csv)           
            csvname , ext = os.path.splitext(csv)
            label=os.path.join(datafol,"label",csvname+"_label"+ext)
            hand=os.path.join(datafol,"hands",csvname+"_hand"+ext)
            
            self.datapath.append([self.name,pts,label,hand,csvname])

        self.seg_classes, self.num_seg_classes=self.name,3
        
        print("self.seg_classes, self.num_seg_classes:",self.seg_classes, self.num_seg_classes)
        

    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set=np.array(pd.read_csv(fn[1],header=None)).astype(np.float32)
        cls = self.classes[self.datapath[index][0]]
        #handを左右で分ける
        
        hand_set = np.array(pd.read_csv(fn[3],header=None)).astype(np.float32)
        label=fn[4]

        hand=np.split(hand_set,2,axis=0)
        #--------変更-------------

        
        hand_l = hand[0]
        hand_r = hand[1]
        #中指の長さが0.5になるように調整する。後に手首座標系に変換するときに0~1になる予想。
        #print(hand_l.shape)
        middle_len =sum(np.array([np.linalg.norm(hand_l[0]-hand_l[8]),
                     np.linalg.norm(hand_l[8]-hand_l[9]),
                     np.linalg.norm(hand_l[9]-hand_l[10]),
                     np.linalg.norm(hand_l[10]-hand_l[20])]))
        
        hand_scale = 1 / middle_len 
        hand_l =hand_l * hand_scale / 2


        #print("RRRRR")
        middle_len =sum(np.array([np.linalg.norm(hand_r[0]-hand_r[8]),
                     np.linalg.norm(hand_r[8]-hand_r[9]),
                     np.linalg.norm(hand_r[9]-hand_r[10]),
                     np.linalg.norm(hand_r[10]-hand_r[20])]))
        
        hscale_r = 1 /middle_len 

        hand_r = hand_r * hscale_r /2
        
        hand_set = np.vstack((hand_l, hand_r))
        #print("shape:", hand_set.shape)

        
        #print(np.linalg.norm(hand_l[0] - hand_l[1],axis=0).shape)
        
        #手首座標系にする?

        
        seg=pd.read_csv(fn[2],header=None).to_numpy().astype(np.int64)
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale
        hand_set_rote = hand_set

        if self.data_augmentation:
            #print("augment")
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            #print(rotation_matrix)
            random_jitter=np.random.normal(0, 0.02, size=point_set.shape)
            point_set[:,[0,1]] = point_set[:,[0,1]].dot(rotation_matrix) # random rotation
            point_set += random_jitter # random jitter
            
            hand_set_rote[:,[0,1]] = hand_set_rote[:,[0,1]].dot(rotation_matrix)
            #hand_set += np.random.normal(0, 0.02, size=hand_set.shape)

        seg = seg[choice]
        batch_weight=np.array([np.count_nonzero(seg==0), np.count_nonzero(seg==1),np.count_nonzero(seg==2)] ) / len(seg)
        point_set = torch.from_numpy(point_set.astype(np.float32))
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        hand_set_rote=torch.from_numpy(hand_set_rote.astype(np.float32))
        batch_weight = torch.from_numpy(batch_weight.astype(np.float32))
        
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg, hand_set_rote,label, batch_weight

     
        #print(point_set[1])
        
  

    def __len__(self):
        return len(self.datapath)
    


class HandDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2048,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split=split

        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        self.name=class_choice[0]
        datafol=os.path.join(self.root,split)
        print(datafol)
        
        self.classes = {self.name:20}
        print(self.classes)
        self.datapath=[]

        
        #print(namefol)
        for csv in os.listdir(datafol):
    
            csvname , ext = os.path.splitext(csv)
            hand=os.path.join(datafol,csvname+ext)
            #print(hand, csvname)
            self.datapath.append([self.name,hand,csvname])
        

    def __getitem__(self, index):
        fn = self.datapath[index]
        #print(fn)
        #handを左右で分ける
        
        hand_set = np.array(pd.read_csv(fn[1],header=None)).astype(np.float32)

        hand=np.split(hand_set,2,axis=0)
        #--------変更-------------

        
        hand_l = hand[0]
        hand_r = hand[1]
        #中指の長さが0.5になるように調整する。後に手首座標系に変換するときに0~1になる予想。
        #print(hand_l.shape)
        middle_len =sum(np.array([np.linalg.norm(hand_l[0]-hand_l[8]),
                     np.linalg.norm(hand_l[8]-hand_l[9]),
                     np.linalg.norm(hand_l[9]-hand_l[10]),
                     np.linalg.norm(hand_l[10]-hand_l[20])]))
        
        hand_scale = 1 / middle_len 
        hand_l =hand_l * hand_scale / 2


        #print("RRRRR")
        middle_len =sum(np.array([np.linalg.norm(hand_r[0]-hand_r[8]),
                     np.linalg.norm(hand_r[8]-hand_r[9]),
                     np.linalg.norm(hand_r[9]-hand_r[10]),
                     np.linalg.norm(hand_r[10]-hand_r[20])]))
        
        hscale_r = 1 /middle_len 

        hand_r = hand_r * hscale_r /2
        
        hand_set = np.vstack((hand_l, hand_r))

        hand_set_rote = hand_set

        if self.data_augmentation:
            #print("augment")
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            hand_set_rote[:,[0,1]] = hand_set_rote[:,[0,1]].dot(rotation_matrix)
            #hand_set += np.random.normal(0, 0.02, size=hand_set.shape)


        hand_set_rote=torch.from_numpy(hand_set_rote.astype(np.float32))
        hand_set=torch.from_numpy(hand_set.astype(np.float32))
        
        
        return hand_set_rote,hand_set

       
  

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    

    def __init__(self,
                 root,
                 npoints=2048,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

