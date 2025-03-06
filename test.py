import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys
 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import test ,seed_torch
from torch.utils.data import DataLoader
from datasets.load_datasets import h5file_Dataset
import torch
import numpy as np
from utilmodule.createmode import create_model





def main(args):
    import pandas as pd
 
    seed_torch(2021)
    res_list = []
    
    basedmodel,ppo,classifymodel,memory,FusionHisF = create_model(args)
    data_csv_dir = args.csv
    h5file_dir = args.test_h5
<<<<<<< HEAD
    test_pth = ''
=======
    test_pth = '/home/ttzheng/MILRLMedical/PAMIL/save_model/Camelyon16_model.pth.tar'
>>>>>>> 9552a20 (Initial commit)
    
    checkpoint = torch.load(test_pth)
    basedmodel.load_state_dict(checkpoint['model_state_dict'])
    FusionHisF.load_state_dict(checkpoint['FusionHisF'])
    classifymodel.load_state_dict(checkpoint['fc'])
    ppo.policy.load_state_dict(checkpoint['policy'])

    basedmodel.eval()
    classifymodel.eval()
    FusionHisF.eval()
    ppo.policy.eval() 

    test_dataset = h5file_Dataset(data_csv_dir,h5file_dir,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
    precision, recall, f1, auc, accuracy = test(args,basedmodel,ppo,classifymodel,FusionHisF,memory,test_dataloader )
    res_list.append([accuracy,auc,precision,recall,f1]) 

    df = pd.DataFrame(res_list, columns=[ 'acc', 'auc', 'precision', 'recall', 'f1'])
    
<<<<<<< HEAD
    df.to_csv('/result.csv', index=False)
=======
    df.to_csv('/home/ttzheng/MILRLMedical/PAMIL/save_model/result.csv', index=False)
>>>>>>> 9552a20 (Initial commit)




if __name__ == "__main__":

    args = make_parse()
    main(args)
