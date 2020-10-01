import sys, os
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=True
import numpy as np
import random
from mtdnn_module import Net
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import torch
import pandas as pd
from rdkit import RDLogger
import heapq
import time
from collections import Counter

class mtdnn_us:
    def __init__(self, model_path:str):
        path_list = [f'mtdnn_pop {i}_128_5e-4.pth' for i in range(5)]
        path_list = [os.path.join(model_path, i) for i in path_list]
        self.mtdnn = [Net() for i in range(5)]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtdnn = [self.mtdnn[i].to(self.device) for i in range(5)]
        self.mtdnn = [torch.nn.DataParallel(self.mtdnn[i]) for i in range(5)]
        for i in range(5):
            self.mtdnn[i].load_state_dict(torch.load(path_list[i]))
    

    def predict(self, smiles:str):
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024)
        act_array = np.zeros((391))
        for i in range(5):
            act_array += self.mtdnn[i](torch.tensor(np.array(fp)).to(self.device).float().unsqueeze(dim=0)).squeeze(dim=0)[0:391].cpu().detach().numpy()
        self.act_list = [list(act_array).index(i) for i in act_array if i>=2.5]
        return self.act_list


    def select_or(self, smiles, class_file:list):
        class_disc = np.zeros([8])
        for j1, j in enumerate(['TK', 'TKL', 'STE', 'CK1', 'AGC', 'CAMK', 'CMGC']):
                class_disc[j1] = [class_file[i] for i in self.predict(smiles)].count(j)
        class_disc[7] = np.sum(class_disc[0:7])

        class_disc_or = np.zeros([7])
        for i in range(7):
            class_disc_or[i] = (class_disc[i] / (class_disc[7] - class_disc[i])) / \
                           ((391 - class_disc[i]) / (391 - class_disc[7] + class_disc[i]))
        return class_disc_or
    
    def spec_or(self, smiles:str, site_list:list, kinase_file:list):
        act_list = self.predict(smiles)
        spec_family = [kinase_file.index(i) for i in site_list]
        family_act = sum([1 for i in spec_family if i in act_list])
        all_act = len(act_list)
        family_or = (family_act / (all_act - family_act))/\
            ((391 - family_act)/(391- all_act + family_act))
        return family_or




        
