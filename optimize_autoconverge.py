import json
import math
import os
import sys
import time
import typing as t
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=True
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
import pandas as pd
from rdkit import RDLogger

import numpy as np
import torch
from torch import nn
from torch import optim



from mtdnn_utils import *
from data_utils import get_data_loader
from data_utils import get_data_loader_full
from mol_spec import MoleculeSpec
from deep_scaffold import DeepScaffold
from eval_sys import *

class Deep_optimizer():
    
    def __init__(self, smiles, type_, model_path):
        #type = family /select
        self.smiles = smiles
        self.type = type_
        self.mtdnn = mtdnn_us(model_path[1])
        self.mdl = build_model(model_path[0])

    def optimize(self, target):
        if self.type == 'family':
            while 1:
                output_mols = sample(self.mdl, self.smiles, 128)
                output_mols_ = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in output_mols[0] if i is not None]
                output_mols_reg = list(set(output_mols_))
                activity = [self.mtdnn.predict(i) for i in output_mols_reg if i is not None]

                kinase_tab = pd.read_csv('./site_info_drawversion.csv')
                classlist = list(kinase_tab.iloc[:, 1])

                kinase_num = ['TK', 'TKL', 'STE', 'CK1', 'AGC', 'CAMK', 'CMGC'].index(target)
                class_disc_or = np.zeros((len(output_mols_reg), 7))
                for i in range(len(output_mols_reg)):
                    class_disc_or[i, :] = self.mtdnn.select_or(output_mols_reg[i], classlist)
               
                count_site = list(class_disc_or[:, kinase_num])
                max_num = heapq.nlargest(16, count_site)
                max_ind = []
                for i in max_num:
                    index = count_site.index(i)
                    max_ind.append(index)
                    count_site[index] = 0
                opt_mol = [Chem.MolFromSmiles(output_mols_reg[i]) for i in max_ind]
                opt_scaffold = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(i)) for i in opt_mol]
                if self.smiles == max(opt_scaffold, key=opt_scaffold.count):
                    break
                else:
                    self.smiles = max(opt_scaffold, key=opt_scaffold.count)
            return [self.smiles, opt_mol]

        if self.type == 'select':
            while 1:
                output_mols = sample(self.mdl, self.smiles, 128)
                output_mols_ = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in output_mols[0] if i is not None]
                output_mols_reg = list(set(output_mols_))

                kinase_tab = pd.read_csv('./site_info_drawversion.csv', index_col=0)
                kinase_list = list(kinase_tab.iloc[:, 0])

                or_list = [self.mtdnn.spec_or(i, target, kinase_list) for i in output_mols_reg]
                max_num = heapq.nlargest(16, or_list)
                max_ind = []
                for i in max_num:
                    index = or_list.index(i)
                    max_ind.append(index)
                    or_list[index] = 0
                opt_mol = [Chem.MolFromSmiles(output_mols_reg[i]) for i in max_ind]
                opt_scaffold = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(i)) for i in opt_mol]
                if self.smiles == max(opt_scaffold, key=opt_scaffold.count):
                    break
                else:
                    self.smiles = max(opt_scaffold, key=opt_scaffold.count)
            return [self.smiles, opt_mol]





