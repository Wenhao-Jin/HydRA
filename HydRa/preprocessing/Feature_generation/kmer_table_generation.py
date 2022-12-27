#!/usr/bin/env python

import pandas as pd
import numpy as np
import multiprocessing as mp
import os

def get_kmer(sequence, k):
    s=sequence.strip(' \n').replace('*','')
    kmers=set([])
    for i in range(len(s)-k+1):
        kmers.add(s[i:i+k])
    
    return kmers

def count_selected_kmers(sequence, protein_name, selected_kmers):
    tmp_kmers_dict={kmer:0 for kmer in selected_kmers}
    k=len(list(selected_kmers)[0])
    s=sequence.strip(' \n').replace('*','')
    for i in range(len(s)-k+1):
        if (s[i:i+k] in selected_kmers):
            tmp_kmers_dict[s[i:i+k]]=tmp_kmers_dict[s[i:i+k]]+1
        
    tmp_kmers_dict['Protein_Name']=protein_name

    return tmp_kmers_dict

def Get_kmer_feature_table(k, protein_files, out_dir, RBP_set, sequence_type='AA', model_name='Model', num_top_kmer_kept=1000):
    kmers_of_RBPs=[]
    sequence_tuple=[]
    for prot_file in protein_files:
        protein_name='.'.join(prot_file.split('/')[-1].split('.')[:-1])
        print(protein_name)
        if prot_file.strip(' ').endswith('.spd3'):
            df_tmp=pd.read_table(prot_file)
            sequence=''.join(list(df_tmp[df_tmp['AA']!='*']['SS']))
        else:
            with open(prot_file) as f:
                sequence=''.join(f.read().strip('* \n').split('\n')[1:])

        sequence_tuple.append((protein_name, sequence))
        if protein_name in RBP_set:
            kmers_of_RBPs.append(get_kmer(sequence, k))

    ##Get 1000 most frequently appeared k-mers in RBPs
    ##Time consuming step
    all_list=[] 
    for kmer in kmers_of_RBPs:
        all_list=all_list+list(kmer)

    all_set=set(all_list)
    kmer_dict={kmer:all_list.count(kmer) for kmer in all_set}

    kmer_df=pd.DataFrame(kmer_dict.items(), columns=['kmer', 'counts']).sort_values('counts', ascending=False)
    selected_kmers=set(kmer_df['kmer'][:num_top_kmer_kept])

    ###multiprocessing approach
    pool=mp.Pool(processes=None)
    results = [pool.apply_async(count_selected_kmers, args=(sequence, protein_name, selected_kmers,)) for protein_name, sequence in sequence_tuple]
    results= [item.get() for item in results if item]
    results = pd.DataFrame(results).set_index('Protein_Name')

    # results=[]
    # for protein_name, sequence in sequence_tuple:
    #     print(count_selected_kmers(sequence, protein_name, selected_kmers))
    #     results.append(count_selected_kmers(sequence, protein_name, selected_kmers))
    #     results = pd.DataFrame(results).set_index('Protein_Name')

    ##annotate RBPs
    results['RBP_flag']=list(map(lambda x: 1 if x in RBP_set else 0, results.index))
    results.to_csv(os.path.join(out_dir, '{}_{}_{}mers_feature_table.txt'.format(model_name,sequence_type,k)), sep='\t', index=True)
