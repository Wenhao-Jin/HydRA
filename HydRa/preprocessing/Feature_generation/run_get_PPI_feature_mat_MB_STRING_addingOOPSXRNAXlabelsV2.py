#!/usr/bin/env python

## V2: update the RBP list by merge combined8RBPlist with the intersection of OOPS and XRNAX RBPs. Rather than take the union of OOPS and XRNAX RBPs.

import networkx as nx
import pandas as pd
import numpy as np
import multiprocessing as mp
import sys
sys.path.insert(0, '/home/wjin/projects/RBP_pred/RBP_identification/scripts/SONAR_for_prediction')
from get_PPI_feature_mat_MB_STRING import get_PPI_features, get_1stPPI_features


# f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/OOPS_XRNAX_novel_RBPs.txt')
# OOPSXRNAX_prots=set(f.read().split('\n'))
# f.close()

G1=nx.read_edgelist('/home/wjin/projects/SONAR_ChernHan/Data/PPI_data_updated/mentha20180108_BioPlex2.0_edgelist.txt')
G1.remove_edges_from(G1.selfloop_edges())
G2=nx.read_edgelist('/home/wjin/projects/SONAR_ChernHan/Data/PPI_data_updated/STRING_v10.5_uniprot_edgelist_withoutExperimentalData.txt')
G2.remove_edges_from(G2.selfloop_edges())

# f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data/RBP_list/Combined8RBPlist_uniprotID_new.txt')
# RBP_merged_set=set(f.read().split('\n'))
# RBP_merged_set.discard('')
# f.close()
# RBP_merged_set=RBP_merged_set.union(OOPSXRNAX_prots)

f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data/RBP_list/Combined8RBPlist_plusOOPSXRNAXsharedRBPs_uniprotID_20190717.txt')
RBP_merged_set=set(f.read().split('\n'))
f.close()

prot_set=set(G1.nodes())
pool=mp.Pool(processes=16)
result=[pool.apply_async(get_PPI_features, args=(prot, G1, RBP_merged_set,)) for prot in prot_set]
results=[p.get() for p in result]
result2=[pool.apply_async(get_1stPPI_features, args=(prot, G2, RBP_merged_set,)) for prot in prot_set]
results2=[p.get() for p in result2]
PPI_feature_table1=pd.DataFrame(results).set_index('Protein_name')
PPI_feature_table2=pd.DataFrame(results2).set_index('Protein_name')
PPI_feature_table=PPI_feature_table1.join(PPI_feature_table2, how='left', rsuffix='_STRING')
PPI_feature_table.to_csv('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/PPI_quality/SONAR_output/menthaBioPlex_separatedSTRING_feature_table_combined8RBPlistAddingOOPSXRNAXlabelsV2.xls', index=True, sep='\t')
