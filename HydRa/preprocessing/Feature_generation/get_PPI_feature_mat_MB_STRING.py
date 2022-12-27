#!/home/wjin/anaconda/envs/wjin/bin/python
## New: compared to old version, this script get three PPI features from experimentally identified data, and get one PPI features from STRING (predicted PPI) data.

import networkx as nx
import pandas as pd
import numpy as np
import multiprocessing as mp

def replace_zeros(d):
    return d if d!=0 else 0.01

def get_1stPPI_features(prot, G, RBP_set):
    #get fisrt-level PPI features
    if prot in G.nodes():
        NBhood1_total=list(G.neighbors(prot))
        RBP1 = len([prot1 for prot1 in NBhood1_total if prot1 in RBP_set])
    else:
        NBhood1_total=[]
        RBP1=0

    NB1=replace_zeros(len(NBhood1_total))
    RBP1_ratio=RBP1*1.0/NB1


    return {'Protein_name':prot, 'RBP_neighbor_counts':RBP1, '1st_neighbor_counts':NB1, 'primary_RBP_ratio':RBP1_ratio}

def get_PPI_features(prot, G, RBP_set):
    #get first-level PPI features
    NBhood1_total=list(G.neighbors(prot))
    NBhood2_total=[]
    NBhood3_total=[]
    for nb1 in NBhood1_total:
        NBhood2=list(G.neighbors(nb1))
        NBhood2.remove(prot)
        NBhood2_total.extend(NBhood2)
        for nb2 in NBhood2:
            NBhood3=list(G.neighbors(nb2))
            NBhood3.remove(nb1)
            NBhood3_total.extend(NBhood3)
            
    NBhood2_total=list(filter(lambda x: x!=prot ,NBhood2_total))
    NBhood3_total=list(filter(lambda x: x!=prot ,NBhood3_total))
    #print NBhood2_total
    #print NBhood3_total
    NBhood1_RBP = [prot1 for prot1 in NBhood1_total if prot1 in RBP_set]
    NBhood2_RBP = [prot2 for prot2 in NBhood2_total if prot2 in RBP_set]
    NBhood3_RBP = [prot3 for prot3 in NBhood3_total if prot3 in RBP_set]
    RBP1=len(NBhood1_RBP)
    RBP2=len(NBhood2_RBP)
    RBP3=len(NBhood3_RBP)
    NB1=replace_zeros(len(NBhood1_total))
    NB2=replace_zeros(len(NBhood2_total))
    NB3=replace_zeros(len(NBhood3_total))
    
    RBP1_ratio=RBP1*1.0/NB1
    RBP2_ratio=RBP2*1.0/NB2
    RBP3_ratio=RBP3*1.0/NB3
    return {'Protein_name':prot, 'RBP_neighbor_counts':RBP1, '1st_neighbor_counts':NB1, 'RBP_2nd_neighbor_counts':RBP2, '2nd_neighbor_counts':NB2, 'RBP_3rd_neighbor_counts':RBP3, '3rd_neighbor_counts':NB3, 'primary_RBP_ratio':RBP1_ratio, 'secondary_RBP_ratio':RBP2_ratio, 'tertiary_RBP_ratio':RBP3_ratio, 'RBP_flag': prot in RBP_set}

def get_PPI_features_SingleCounting(prot, G, RBP_set):
    #get first-level PPI features
    NBhood1_total=G.neighbors(prot)
    NBhood2_total=[]
    NBhood3_total=[]
    for nb1 in NBhood1_total:
        NBhood2=G.neighbors(nb1)
        NBhood2.remove(prot)
        NBhood2_total.extend(NBhood2)
        for nb2 in NBhood2:
            NBhood3=G.neighbors(nb2)
            NBhood3.remove(nb1)
            NBhood3_total.extend(NBhood3)
            
    NBhood2_total=filter(lambda x: x!=prot ,NBhood2_total)
    NBhood3_total=filter(lambda x: x!=prot ,NBhood3_total)
    #print NBhood2_total
    #print NBhood3_total
    NBhood1_RBP = [prot1 for prot1 in NBhood1_total if prot1 in RBP_set]
    NBhood2_RBP = [prot2 for prot2 in NBhood2_total if prot2 in RBP_set]
    NBhood3_RBP = [prot3 for prot3 in NBhood3_total if prot3 in RBP_set]
    RBP1=len(set(NBhood1_RBP))
    RBP2=len(set(NBhood2_RBP))
    RBP3=len(set(NBhood3_RBP))
    NB1=replace_zeros(len(set(NBhood1_total)))
    NB2=replace_zeros(len(set(NBhood2_total)))
    NB3=replace_zeros(len(set(NBhood3_total)))
    
    RBP1_ratio=RBP1*1.0/NB1
    RBP2_ratio=RBP2*1.0/NB2
    RBP3_ratio=RBP3*1.0/NB3
    return {'Protein_name':prot, 'RBP_neighbor_counts':RBP1, '1st_neighbor_counts':NB1, 'RBP_2nd_neighbor_counts':RBP2, '2nd_neighbor_counts':NB2, 'RBP_3rd_neighbor_counts':RBP3, '3rd_neighbor_counts':NB3, 'primary_RBP_ratio':RBP1_ratio, 'secondary_RBP_ratio':RBP2_ratio, 'tertiary_RBP_ratio':RBP3_ratio, 'RBP_flag': prot in RBP_set}


def get_PPI_features2(G, RBP_set):
    #get first-level PPI features
    results=[]
    for prot in G.nodes_iter():
        NBhood1_total=G.neighbors(prot)
        NBhood2_total=[]
        NBhood3_total=[]
        for nb1 in NBhood1_total:
            NBhood2=G.neighbors(nb1)
            NBhood2.remove(prot)
            NBhood2_total.extend(NBhood2)
            for nb2 in NBhood2:
                NBhood3=G.neighbors(nb2)
                NBhood3.remove(nb1)
                NBhood3_total.extend(NBhood3)

        NBhood2_total=filter(lambda x: x!=prot ,NBhood2_total)
        NBhood3_total=filter(lambda x: x!=prot ,NBhood3_total)
        #print NBhood2_total
        #print NBhood3_total
        NBhood1_RBP = [prot1 for prot1 in NBhood1_total if prot1 in RBP_set]
        NBhood2_RBP = [prot2 for prot2 in NBhood2_total if prot2 in RBP_set]
        NBhood3_RBP = [prot3 for prot3 in NBhood3_total if prot3 in RBP_set]
        RBP1=len(NBhood1_RBP)
        RBP2=len(NBhood2_RBP)
        RBP3=len(NBhood3_RBP)
        NB1=replace_zeros(len(NBhood1_total))
        NB2=replace_zeros(len(NBhood2_total))
        NB3=replace_zeros(len(NBhood3_total))

        RBP1_ratio=RBP1*1.0/NB1
        RBP2_ratio=RBP2*1.0/NB2
        RBP3_ratio=RBP3*1.0/NB3
        results.append({'Protein_name':prot, 'RBP_neighbor_counts':RBP1, '1st_neighbor_counts':NB1, 'RBP_2nd_neighbor_counts':RBP2, '2nd_neighbor_counts':NB2, 'RBP_3rd_neighbor_counts':RBP3, '3rd_neighbor_counts':NB3, 'primary_RBP_ratio':RBP1_ratio, 'secondary_RBP_ratio':RBP2_ratio, 'tertiary_RBP_ratio':RBP3_ratio})
        
    return results

#results=get_PPI_features2(G, RBP_merged_set)
#PPI_feature_table=pd.DataFrame(results).set_index('Protein_name')

if __name__=='__main__':
    G=nx.read_edgelist('/home/wjin/projects/RBP_pred/RBP_identification/Data/BioPlex_interactionList_v4_edgelist.txt')
    G.remove_edges_from(G.selfloop_edges())
    newRBP_table=pd.read_table('/home/wjin/Dropbox/RBP_identification_Kris_Wenhao/merged_list_figures/Lists/merged_RBP_list.xls', sep='\t', index_col=0)
    RBP_merged_set=set(newRBP_table.iloc[:,0])
    prot_set=set(G.nodes())
    pool=mp.Pool(processes=16)
    result=[pool.apply_async(get_PPI_features, args=(prot, G, RBP_merged_set,)) for prot in prot_set]
    results=[p.get() for p in result]
    PPI_feature_table=pd.DataFrame(results).set_index('Protein_name')
    PPI_feature_table.to_csv('/home/wjin/projects/RBP_pred/RBP_identification/Data/PPI_feature_label_table_NewRBPlist_v4.xls', index=True, sep='\t')
