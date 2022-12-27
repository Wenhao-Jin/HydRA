#!/usr/bin/env python
import pandas as pd
import numpy as np
import multiprocessing as mp
import os

def calculate_AAC(seq):
    AAs=['H','K','D','E','S','T','N','Q','C','G','P','A','V','I','L','M','F','Y','W','R']
    dic = {aa:0 for aa in AAs}
    seq_len=len(seq)
    for aa in seq:
        dic[aa]+=1

    return {k:v*1.0/seq_len for k, v in dic.items()}

def get_AAC_features(prot_file):
    f=open(prot_file, 'r')
    s=''.join(f.read().strip('* \n').split('\n')[1:])
    f.close()
    s=s.upper()
    s=s.replace('*','')
    s=s.replace('X','') #solve bug for those cannot be processed by propy
    s=s.replace('U','C') #solve bug for those 'U' that cannot be processed by propy
    try:
        aac=calculate_AAC(s)  #the output is a dictionary
        aac['Protein_Name']=prot_file.split('/')[-1].split('.')[0]
    except Exception as e:
        print(type(e))
        print(e)
        print(prot_file.split('/')[-1].split('.')[0])
        return None

    return aac

def get_AAC_features_bk(prot_file):
    f=open(prot_file, 'r')
    s=''.join(f.read().strip('* \n').split('\n')[1:])
    f.close()
    s=s.upper()
    s=s.replace('*','')
    s=s.replace('X','') #solve bug for those cannot be processed by propy
    s=s.replace('U','C') #solve bug for those 'U' that cannot be processed by propy
    try:
        DesObject=PyPro.GetProDes(s)
        aac=DesObject.GetPAAC(lamda=0, weight=0.05)  #the output is a dictionary
        aac['Protein_Name']=prot_file.split('/')[-1].split('.')[0]
    except Exception as e:
        print(type(e))
        print(e)
        print(prot_file.split('/')[-1].split('.')[0])
        return None

    return aac
    

if __name__=='__main__':
	###multiprocessing approach
	t_dir='/home/wjin/data2/proteins/WuhanVirus/proteins/' 
	prot_files=set(filter(lambda x:x.endswith('.fasta'), os.listdir(t_dir)))

	pool1=mp.Pool(processes=8)
	results1 = [pool1.apply(get_AAC_features, args=(os.path.join(t_dir, prot), 4)) for prot in prot_files if os.path.exists(os.path.join(t_dir, prot))]
	results1=[item.get() for item in results1 if item]
	results1 = pd.DataFrame(results1).set_index('Protein_Name')
	results1.to_csv('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/AAC_feature_table_WuhanVirus.txt', sep='\t', index=True)

