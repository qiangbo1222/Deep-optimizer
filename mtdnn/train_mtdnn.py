from __future__ import print_function
import numpy as np
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

clusterpath = './inputs.h5'
fold_dictpath = './folddict.csv'
labels_path = './labels.h5'
weight_path = './weight_metrix.h5'

class netdata(Dataset): 
    def __init__(self, clusterpath, fold_dictpath, label_path, weight_path, fold_idx):
        tmp = np.array(pd.read_csv(fold_dictpath).iloc[fold_idx]).reshape(1, -1)
        fdict = tmp[~np.isnan(tmp)]
        self.weight_metrix = pd.read_hdf(weight_path, 'key').loc[fdict]
        self.fold_data = pd.read_hdf(clusterpath, 'key').loc[fdict]
        self.label_data = pd.read_hdf(label_path, 'key').loc[fdict]
    def __len__(self):
        return self.label_data.shape[0]
    def __getitem__(self, idx):
        input_tensor = torch.tensor(np.array(self.fold_data.iloc[idx]))
        label_tensor = torch.tensor(np.array(self.label_data.iloc[idx]))
        weight_tensor = torch.tensor(np.array(self.weight_metrix.iloc[idx]))
        return input_tensor, label_tensor, weight_tensor





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hid1 = nn.Linear(1024, 1024)
        self.hid1_drop = nn.Dropout(0.5)
        self.hid2 = nn.Linear(1024, 391)
        self.hid2_drop = nn.Dropout(0.5)
        self.out = nn.Linear(391,782)

    def forward(self,x):
        x = F.relu(self.hid1_drop(self.hid1(x)))
        x = F.relu(self.hid2_drop(self.hid2(x)))
        x = F.relu(self.out(x))
        x = x.view(x.shape[0], 2, 391)
        x = F.softmax(x, dim=1)
        x = x.view(x.shape[0], 782)
        return x







all_roc = []
for  i in range(5):
    loss = 0.
    roc_count = 0
    au_roc=[]
    list_fold = [0,1,2,3,4]
    list_fold.remove(i)
    
    mtdnn = Net()
    mtdnn = torch.nn.DataParallel(mtdnn)
    mtdnn = mtdnn.to(device)
    #optimizer = optim.SGD(mtdnn.parameters(), weight_decay=0.002, lr=5e-4)
    optimizer=torch.optim.Adam(mtdnn.parameters(), lr=5e-4, betas=(0.9,0.999), weight_decay=0.002)

    train_data = netdata(clusterpath, fold_dictpath, labels_path, weight_path, list_fold)
    data_load = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0 )
    test_data = netdata(clusterpath, fold_dictpath, labels_path, weight_path, i)

    for epoch in range(500000):
        for step, batch in enumerate(data_load):
            input_batch, label_batch, weight_batch = batch[0], batch[1], batch[2]
            input_batch, label_batch, weight_batch =  input_batch.to(device).float(), label_batch.to(device).float(), weight_batch.to(device).float()
            optimizer.zero_grad()
            outputs = mtdnn(input_batch)
            
            
            
            label_batch = torch.cat((label_batch, 1-label_batch), dim=1)
            weight_batch = torch.cat((weight_batch, weight_batch), dim=1)
            
            
            
            loss = F.binary_cross_entropy(outputs, label_batch, weight=weight_batch)
            loss.backward()
            optimizer.step()
        print(f'\rtask {i} epoch {epoch} running', end=' ')
            
        test_score_list = torch.zeros([len(test_data),782])
        test_label_list = torch.zeros([len(test_data),391])
        if epoch%5 == 0:
            for step, test_datas in enumerate(test_data):
                input_ecfp4, test_label_list[step] = test_datas[0].to(device).float(), test_datas[1].to(device).float() 
                input_ecfp4 = input_ecfp4.unsqueeze(0)
                test_score_list[step] = mtdnn(input_ecfp4).squeeze(0)
            test_label = test_label_list.view(-1)
            temp = test_score_list[:,0:391]
            test_score = temp.reshape(len(test_label))
            fpr, tpr, _ = metrics.roc_curve(test_label.cpu().detach().numpy(), test_score.cpu().detach().numpy(),  pos_label=1 )
            au_roc.append(metrics.auc(fpr, tpr))
            
            if epoch>=10: 
                if au_roc[-1] < au_roc[-2]:
                    roc_count += 1
                    if roc_count == 4:
                        break
                else:
                    roc_count=0
            
            print(' \r task %d training at epoch %d now have auroc %f '%(i, epoch, au_roc[-1]), end='')
 
    all_roc.append(au_roc[-1])
    model_PATH = './models_15_decay/mtdnn_pop %d_128_5e-4.pth'%(i)
    torch.save(mtdnn.state_dict(), model_PATH)

                    

print(' finish mtdnn training ', end= ' ')
pd.DataFrame(all_roc).to_csv('./models_15_decay/5 auroc.csv')
