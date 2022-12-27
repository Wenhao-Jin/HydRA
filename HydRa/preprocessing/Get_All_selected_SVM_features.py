#!/usr/bin/env python
### Same as Get_All_selected_SVM_features.py
import pandas as pd
import numpy as np
import multiprocessing as mp
import os
#from propy import PyPro


def count_selected_aakmers(prot_file, selected_kmers):
    print(prot_file)
    tmp_kmers_dict={kmer:0 for kmer in selected_kmers}
    k=len(next(iter(selected_kmers)))
    f=open(prot_file, 'r')
    s=''.join(f.read().strip('* \n').split('\n')[1:])
    f.close()
    s=s.replace('*','')
    for i in range(len(s)-k+1):
#        print len(s[i:i+k])
        if (s[i:i+k] in selected_kmers):
#            print "yes"
            tmp_kmers_dict[s[i:i+k]]+=1
        
    tmp_kmers_dict['Protein_Name']=prot_file.split('/')[-1].split('.')[0]
#    print tmp_kmers_dict['RLPL']
    return tmp_kmers_dict

def count_selected_SSkmers(prot_spd3_file, selected_kmers):
    print(prot_spd3_file)
    tmp_kmers_dict={kmer:0 for kmer in selected_kmers}
    k=len(next(iter(selected_kmers)))
    s=''.join(list(pd.read_table(prot_spd3_file)['SS']))
    for i in range(len(s)-k+1):
#        print len(s[i:i+k])
        if (s[i:i+k] in selected_kmers):
#            print "yes"
            tmp_kmers_dict[s[i:i+k]]=tmp_kmers_dict[s[i:i+k]]+1
        
    tmp_kmers_dict['Protein_Name']=prot_spd3_file.split('/')[-1].split('.')[0]
#    print tmp_kmers_dict['RLPL']
    return tmp_kmers_dict

def Get_feature_table(protein_set, aaSeq_dir, ssSeq_dir, PseAAC_filepath, out_dir, job_name):
    f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_mer3_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
    selected_aa3mers=f.read().split('\n')
    f.close()
    f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_mer4_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
    selected_aa4mers=f.read().split('\n')
    f.close()
    f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SS_11mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
    selected_SS11mers=f.read().split('\n')
    f.close()
    f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SS_15mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
    selected_SS15mers=f.read().split('\n')
    f.close()
    # f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_PseAAC_chi2_alpha0.01_selected_features_From_WholeDataSet.txt')
    # selected_PseAAC=f.read().split('\n')
    # f.close()
    f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_AAC_chi2_alpha0.01_selected_features_From_WholeDataSet_AACname.txt')
    selected_PseAAC=f.read().split('\n')
    selected_PseAAC.remove('')
    f.close()
    
    #### for-loop approach
    aa3mer=[]
    aa4mer=[]
    ss11mer=[]
    ss15mer=[]
    protein_set=[prot for prot in protein_set if os.path.exists(os.path.join(aaSeq_dir, prot+'.spd3'))]
    for prot in protein_set:
        aa3mer.append(count_selected_aakmers(os.path.join(aaSeq_dir, prot+'.fasta'), selected_aa3mers))
        aa4mer.append(count_selected_aakmers(os.path.join(aaSeq_dir, prot+'.fasta'), selected_aa4mers))
        ss11mer.append(count_selected_SSkmers(os.path.join(ssSeq_dir, prot+'.spd3'), selected_SS11mers))
        ss15mer.append(count_selected_SSkmers(os.path.join(ssSeq_dir, prot+'.spd3'), selected_SS15mers))


    aa_df1 = pd.DataFrame(aa3mer).set_index('Protein_Name')
    aa_df2 = pd.DataFrame(aa4mer).set_index('Protein_Name')
    ss_df1 = pd.DataFrame(ss11mer).set_index('Protein_Name')
    ss_df2 = pd.DataFrame(ss15mer).set_index('Protein_Name')

    aac_df=pd.read_table(PseAAC_filepath,index_col=0)
    aac_df=aac_df.loc[:,selected_PseAAC]
    aac_df=aac_df.fillna(value=0)


    # ###multiprocessing approach
    # pool=mp.Pool(processes=8)
    # results1 = [pool.apply(count_selected_aakmers, args=(os.path.join(aaSeq_dir, prot+'.fasta'), selected_aa3mers)) for prot in protein_set if os.path.exists(os.path.join(aaSeq_dir, prot+'.fasta'))]
    # aa_df1 = pd.DataFrame(results1).set_index('Protein_Name')

    # pool=mp.Pool(processes=8)
    # results2 = [pool.apply(count_selected_aakmers, args=(os.path.join(aaSeq_dir, prot+'.fasta'), selected_aa4mers)) for prot in protein_set if os.path.exists(os.path.join(aaSeq_dir, prot+'.fasta'))]
    # aa_df2 = pd.DataFrame(results2).set_index('Protein_Name')

    # pool=mp.Pool(processes=8)
    # results3 = [pool.apply(count_selected_SSkmers, args=(os.path.join(ssSeq_dir, prot+'.spd3'), selected_SS11mers)) for prot in protein_set if os.path.exists(os.path.join(ssSeq_dir, prot+'.spd3'))]
    # ss_df1 = pd.DataFrame(results3).set_index('Protein_Name')

    # pool=mp.Pool(processes=8)
    # results4 = [pool.apply(count_selected_SSkmers, args=(os.path.join(ssSeq_dir, prot+'.spd3'), selected_SS15mers)) for prot in protein_set if os.path.exists(os.path.join(ssSeq_dir, prot+'.spd3'))]
    # ss_df2 = pd.DataFrame(results4).set_index('Protein_Name')

    # aac_df=pd.read_table(PseAAC_filepath,index_col=0)
    # aac_df=aac_df[selected_PseAAC]

    results=aa_df1.join(aa_df2, how='inner').join(ss_df1, how='inner').join(ss_df2, how='inner').join(aac_df,how='inner')
    #rf_df=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/selected/SVM_SeqFeature_table_comined_all_selected_features_From_WholeDataset.txt',index_col=0).drop('RBP_flag',axis=1)
    #results=results[rf_df.columns]
    results.to_csv(os.path.join(out_dir,'SVM_seq_features'+job_name+'.xls'), sep='\t', index=True)

if __name__=='__main__':
	t_dir='/home/wjin/data2/proteins/WuhanVirus/proteins/' 
	proteins=map(lambda x: '.'.join(x.split('.')[:-1]), set(filter(lambda x:x.endswith('.fasta'), os.listdir(t_dir))))
	aaSeq_dir = t_dir
	ssSeq_dir = t_dir
	PseAAC_filepath = '/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/AAC_feature_table_WuhanVirus.txt'
	out_dir ='/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_feature/'
	Get_feature_table(proteins, aaSeq_dir, ssSeq_dir, PseAAC_filepath, out_dir, 'WuhanVirus')


    
