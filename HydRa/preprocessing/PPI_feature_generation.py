#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import networkx as nx

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

def get_PPI_features(prot, G, RBP_set, PPI_1stNBs=None):
    """
    G: networkx.Graph object.
    RBP_set: a set of RBP names, used for labeling proteins.
    PPI_1stNBs: list of protein ID or names, if the 1st neighborhoods of the given proteins are also provided.
    """
    #get first-level PPI features
    if PPI_1stNBs:
        NBhood1_total=[p for p in PPI_1stNBs if p in G]
    else:
        NBhood1_total=list(G.neighbors(prot))
    NBhood2_total=[]
    NBhood3_total=[]
    for nb1 in NBhood1_total:
        NBhood2=list(G.neighbors(nb1))
        if prot in NBhood2:
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

def get_PPI_feature_vec(prot, G, RBP_set, num_cut=5, PPI_1stNBs=None):
    #print(prot)
    PPI_features=get_PPI_features(prot, G, RBP_set, PPI_1stNBs)
    if PPI_features['1st_neighbor_counts']>num_cut:
        PPI_features['Reliability']=1
    else:
        PPI_features['Reliability']=-1

    return np.array([PPI_features[k] for k in ['primary_RBP_ratio','secondary_RBP_ratio','tertiary_RBP_ratio','Reliability']])

def get_1stPPI_feature_vec(prot, G, RBP_set, num_cut=5):
    #print(prot)
    if prot in G:
        PPI_features=get_1stPPI_features(prot, G, RBP_set)
    else:
        PPI_features={'Protein_name':prot, 'RBP_neighbor_counts':0, '1st_neighbor_counts':0, 'primary_RBP_ratio':0}
    if PPI_features['1st_neighbor_counts']>num_cut:
        PPI_features['Reliability']=1
    else:
        PPI_features['Reliability']=-1

    return np.array([PPI_features[k] for k in ['primary_RBP_ratio','Reliability']])



if __name__=='__main__':
    G1=nx.read_edgelist('/home/wjin/projects/SONAR_ChernHan/Data/PPI_data_updated/mentha20180108_BioPlex2.0_edgelist.txt')
    #G1=nx.read_edgelist('/home/wjin/projects/SONAR_ChernHan/Data/PPI_data_updated/mentha_human_20200217_edgelist_UniprotID.csv',delimiter=',', data=(('Score',float),))
    G1.remove_edges_from(G1.selfloop_edges())
    f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data/RBP_list/Combined8RBPlist_plusOOPSXRNAXsharedRBPs_uniprotID_20190717.txt')
    RBP_merged_set=set(f.read().split('\n'))
    f.close()
    xls0=pd.ExcelFile('/home/wjin/projects/Coronavirus/data/Coronavirus_PPI_media-4.xlsx')
    interactors_df=xls0.parse(sheet_name=xls0.sheet_names[0], header=[0,1])['Bait-Prey Information']
    proteins=set(interactors_df['Bait'])
    interactors_dic={prot:list(interactors_df[interactors_df.Bait==prot]['Preys']) for prot in proteins}
    num_cut=5 ## cutoff for the size 1st-neighborhood 
    model_PPI=joblib.load('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_models/SONAR_plus_menthaBioPlexSTRING/SVM_PPI_MB_trainedWithWholeDataset_menthaBioPlexSTRING_Final_Model_v4_7_3_MenthaBioPlex_STRING_AddingOOPSXRNAXlabels_newSklearn0.22.1_ModelFile.pkl')
    model_PPI.kernel='rbf'

    HydRa_PPI_predictions={}
    for prot in proteins:
        PPI_features_vec=get_PPI_feature_vec(prot, G1, RBP_merged_set, interactors_dic[prot])
        print(PPI_features_vec)
        HydRa_PPI_predictions[prot]=model_PPI.predict_proba([PPI_features_vec])[:,1][0]

    out_df=pd.DataFrame.from_dict(HydRa_PPI_predictions, orient='index', columns=['HydRa_PPI_score (thre0.67)'])
    out_df
    out_df.to_csv('./data/Baits_HydRaPPI_prediction_menthaBioPlex.tsv')

