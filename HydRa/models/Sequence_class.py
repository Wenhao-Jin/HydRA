#!/usr/bin/env python

import pandas as pd
import numpy as np
import os

aa_code={'H':1, 'K':2, 'D':3, 'E':4,
                 'S':5, 'T':6, 'N':7, 'Q':8, 'C':9,
                 'U':10, 'G':11, 'P':12, 'A':13, 'V':14,
                 'I':15, 'L':16, 'M':17, 'F':18, 'Y':19,
                 'W':20, 'R':21, 'blank':0}
aa_code_reverse={v:k for k, v in aa_code.items()}
# BioVec_weights=pd.read_table('/home/wjin/projects/RBP_pred/RBP_identification/Data/protVec_100d_3grams.csv', sep='\t', header=None, index_col=0)
# BioVec_weights_add_null=np.append(np.zeros((1,100)), BioVec_weights.values, axis=0) #append a [0,0,...,0] array at the top of the matrix, which used for padding 0s.
# BioVec_weights_add_null=BioVec_weights_add_null*10
# BioVec_name_dict={}
# for i in range(1, len(BioVec_weights)+1):
#     BioVec_name_dict.update({BioVec_weights.index[i-1]:i})


class Protein_Sequence_Input5:
    """
    Modified so that when doing the padding, always pad the vector to the length of (self.max_seq_len/(maxlen/2))*(maxlen/2)+self.maxlen/2, which is consistent with the length used in DNNseq class.
    Different from version Protein_Sequence_Input4: Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding amino acid (from aa3mer)
    """
    def __init__(self, files, class_labels, BioVec_name_dict, maxlen=1500):
        """
        files: a list of sequence filenames including the absolute path, best in numpy.array format.
        class_labels: a list recording the RBP identity (True or False) of each protein mentioned in "files", best in numpy.array format.
        PPI_feature_vectors: a list of PPI feature vectors for each protein mentioned in "files", best in numpy.array format.
        BioVec_name_dict: dictionary, a dictory telling the mapping between the aa triplet (e.g, HKE) and their index in this analysis.
        maxlen: The original window size of the DNNseq model.
        max_seqlen: int, the maximum length of amino acids that can be retained.
        """
        if len(files)!=len(class_labels) :
            raise ValueError('The length of files list and class_labels list should be same.')
            
        self.aa_code={'H':1, 'K':2, 'D':3, 'E':4,
                 'S':5, 'T':6, 'N':7, 'Q':8, 'C':9,
                 'U':10, 'G':11, 'P':12, 'A':13, 'V':14,
                 'I':15, 'L':16, 'M':17, 'F':18, 'Y':19,
                 'W':20, 'R':21}
        self.ss_code={'E':1, 'H':2, 'C':3}
        self.ss_sparse_code={'E':[1,0,0], 'H':[0,1,0], 'C':[0,0,1]}
        self.ss_mer_code={'EEE':1, 'HHH':2, 'CCC':3,
                          'EEH':4, 'EEC':5,'EHE':6,'EHH':7,'EHC':8,'ECE':9,'ECH':10,'ECC':11,
                          'HEE':12,'HEH':13,'HEC':14,'HHE':15,'HHC':16,'HCE':17,'HCH':18,'HCC':19,
                          'CEE':20,'CEH':21,'CEC':22,'CHE':23,'CHH':24,'CHC':25,'CCE':26,'CCH':27} #use number 28 to represent unknown 3-mers
        
        self.max_seq_len=maxlen
        self.maxlen=maxlen
        self.BioVec_name_dict=BioVec_name_dict
        self.BioVec_name_keys=self.BioVec_name_dict.keys()
        self.prot_names=[]
        self.seqs=[]
        self.seqlen=[]
        #self.PseAAC_mat=[]
        self.seq_mats=[]
        self.ss_mats=[]
        self.ss_sparse_mats=[]
        self.ss_sparse_mats2=[] #Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding aa3mer
        self.aa_ss_mixed_mats=[]
        self.prot_3mer_mats=[]
        self.ss_3mer_mats=[]
        #self.seq_train_mats=[]
        #self.seq_val_mats=[]
        self.labels=[]
        self.train_labels=[]
        self.val_labels=[]
        self.usable_files=[]
        self.batch_id=0
        #max_seq=0
        for seq_file, class_label in zip(files, class_labels):
            try:
                seq, ss_seq, seq_len = self.get_sequence(seq_file)
                self.seqs.append((seq, ss_seq))
                self.seqlen.append(seq_len)
                self.labels.append(class_label)
                self.usable_files.append(seq_file)
                if seq_len>self.max_seq_len:
                    self.max_seq_len=seq_len
                    #max_seq=seq_file

            except ValueError:
                print(seq_file+" is ignored, because of the length conflict in seq_file and ss_seq_file.")
        
        #print max_seq
        
        if len(self.seqs)!=len(self.labels):
            raise ValueError('The length of generated seq vector and class_labels list should be same.')
            
        print("The maximum length of the sequences is : "+str(self.max_seq_len))

        self.padding_maxlen=int((self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+self.maxlen/2)
        
        for seq, ss_seq in self.seqs:
            seq_mat, ss_mat, aa_ss_mixed_mat, prot_3mer_mat, ss_3mer_mat, ss_sparse_mat, ss_sparse_mat2 = self.encode_protein(seq, ss_seq)
            self.seq_mats.append(seq_mat)
            self.ss_mats.append(ss_mat)
            self.aa_ss_mixed_mats.append(aa_ss_mixed_mat)
            self.prot_3mer_mats.append(prot_3mer_mat)
            self.ss_3mer_mats.append(ss_3mer_mat)
            self.ss_sparse_mats.append(ss_sparse_mat)
            self.ss_sparse_mats2.append(ss_sparse_mat2)
            
        #TODO: Split into training, validation and test dataset
        self.seq_mats=np.array(self.seq_mats)
        self.ss_mats=np.array(self.ss_mats)
        self.ss_sparse_mats=np.array(self.ss_sparse_mats)
        self.ss_sparse_mats2=np.array(self.ss_sparse_mats2)
        self.aa_ss_mixed_mats=np.array(self.aa_ss_mixed_mats)
        self.prot_3mer_mats=np.array(self.prot_3mer_mats)
        self.ss_3mer_mats=np.array(self.ss_3mer_mats)
        #self.PseAAC_mat=np.array(self.PseAAC_mat)
        self.seqlen=np.array(self.seqlen)
        self.labels=np.array(self.labels)
        self.usable_files=np.array(self.usable_files)

        
    def get_sequence(self, seq_file):
        #print seq_file
        f=open(seq_file, 'r')
        s=f.read()
        try:
            s=''.join(s.strip('* \n').split('\n')[1:])
            s=s.replace('*','')
            s=s.replace('X','')
        except:
            print("invalid sequence file:"+seq_file)
            f.close()
            return None, None
            
        f.close()
        if os.path.exists(seq_file.replace('.fasta', '.spd3')):
            seq_file2=seq_file.replace('.fasta', '.spd3')
            tmp=pd.read_table(seq_file2)
            try:
                tmp=tmp[tmp['AA']!='*']
                tmp=tmp[tmp['AA']!='X']
                ss=''.join(list(tmp['SS']))
            except:
                raise ValueError('Got error when reading {}.'.format(seq_file2))
        elif os.path.exists(seq_file.replace('.fasta', '.txt')):
            seq_file2=seq_file.replace('.fasta', '.txt')
            with open(seq_file2) as f:
                ss=''.join(s.strip('* \n').split('\n')[1:])
        else:
            raise FileNotFoundError('Neither {} nor {} is found.'.format(seq_file.replace('.fasta', '.spd3'), seq_file.replace('.fasta', '.txt')))
                    
        if len(s)!=len(ss):
            raise ValueError('The length of sequence and SS sequence is different. Files: '+seq_file+', '+seq_file2+'.')
        

        return s, ss, len(s)
    
    def encode_amino_acid(self, aa): #convert the AA to a integer, rather than a binary vector.
        aa=aa.upper()
        v=self.aa_code[aa]
        return v

    def encode_secondary_structure(self, ss):
        ss=ss.upper()
        v=self.ss_code[ss]
        return v
    
    def encode_secondary_structure_sparse(self, ss):
        ss=ss.upper()
        v=self.ss_sparse_code[ss]
        return v
    
    def encode_aa_3mer(self, mer):
        mer=mer.upper()
        if mer in self.BioVec_name_keys:
            v=self.BioVec_name_dict[mer]
        else:
            v=9048 # use the index of '<unk>' from the BioVec weights matrix.
        return v
    
    def encode_ss_3mer(self, mer):
        mer=mer.upper()
        if mer in self.ss_mer_code:
            v=self.ss_mer_code[mer]
        else:
            v=28
        return v
    
    def encode_protein(self, prot_seq, ss_seq):
        """
        return: protein_sequence_using_digits, secondary_structure_sequence_using_digits, mixed_aa_ss_using_digits, protein_3mer_sequence_using_digits
        """
        length=len(prot_seq)
        prot_seq_l=[self.encode_amino_acid(aa) for aa in prot_seq]
        prot_seq_l+=[0]*(self.padding_maxlen-length)
        ss_seq_l=[self.encode_secondary_structure(ss) for ss in ss_seq]
        ss_seq_l+=[0]*(self.padding_maxlen-length)
        ss_seq_sparse_l=[self.encode_secondary_structure_sparse(ss) for ss in ss_seq]
        ss_seq_sparse2_l=ss_seq_sparse_l[1:-1]
        ss_seq_sparse_l+=[[0,0,0]]*(self.padding_maxlen-length)
        ss_seq_sparse2_l+=[[0,0,0]]*(self.padding_maxlen-length)
        
        prot_3mer_seq_l=[]
        for i in range(len(prot_seq)-2):
            prot_3mer_seq_l.append(self.encode_aa_3mer(prot_seq[i:i+3]))
        
        #prot_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        prot_3mer_seq_l+=[0]*(self.padding_maxlen-(length))
        
        ss_3mer_seq_l=[]
        for i in range(len(ss_seq)-2):
            ss_3mer_seq_l.append(self.encode_ss_3mer(ss_seq[i:i+3]))
        
        #ss_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        ss_3mer_seq_l+=[0]*(self.padding_maxlen-(length))


        #return csr_matrix(np.array(seq_l))
        return np.array(prot_seq_l), np.array(ss_seq_l), np.array(zip(prot_seq_l, np.array(ss_seq_l)+21)).flatten(), np.array(prot_3mer_seq_l), np.array(ss_3mer_seq_l), np.array(ss_seq_sparse_l), np.array(ss_seq_sparse2_l)
        
    
    def get_aa3mer_mats(self):
        return self.prot_3mer_mats

    def get_ss3mer_mats(self):
        return self.ss_3mer_mats
    
    def get_ss_sparse_mats(self):
        return self.ss_sparse_mats

    def get_ss_sparse_mats2(self): #Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding aa3mer
        return self.ss_sparse_mats2
    
    def get_class_labels(self):
        return self.labels

    def get_seqlens(self):
        return self.seqlen

    def get_usable_files(self):
        return self.usable_files

    def get_maxlen(self):
        return self.max_seq_len

class Protein_Sequence_Input5_noSS:
    """
    Modified so that when doing the padding, always pad the vector to the length of (self.max_seq_len/(maxlen/2))*(maxlen/2)+self.maxlen/2, which is consistent with the length used in DNNseq class.
    Different from version Protein_Sequence_Input4: Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding amino acid (from aa3mer)
    """
    def __init__(self, files, class_labels, BioVec_name_dict, maxlen=1500):
        """
        files: a list of sequence filenames including the absolute path, best in numpy.array format.
        class_labels: a list recording the RBP identity (True or False) of each protein mentioned in "files", best in numpy.array format.
        PPI_feature_vectors: a list of PPI feature vectors for each protein mentioned in "files", best in numpy.array format.
        BioVec_name_dict: dictionary, a dictory telling the mapping between the aa triplet (e.g, HKE) and their index in this analysis.
        maxlen: The original window size of the DNNseq model.
        max_seqlen: int, the maximum length of amino acids that can be retained.
        """
        if len(files)!=len(class_labels) :
            raise ValueError('The length of files list and class_labels list should be same.')
            
        self.aa_code={'H':1, 'K':2, 'D':3, 'E':4,
                 'S':5, 'T':6, 'N':7, 'Q':8, 'C':9,
                 'U':10, 'G':11, 'P':12, 'A':13, 'V':14,
                 'I':15, 'L':16, 'M':17, 'F':18, 'Y':19,
                 'W':20, 'R':21}
        # self.ss_code={'E':1, 'H':2, 'C':3}
        # self.ss_sparse_code={'E':[1,0,0], 'H':[0,1,0], 'C':[0,0,1]}
        # self.ss_mer_code={'EEE':1, 'HHH':2, 'CCC':3,
        #                   'EEH':4, 'EEC':5,'EHE':6,'EHH':7,'EHC':8,'ECE':9,'ECH':10,'ECC':11,
        #                   'HEE':12,'HEH':13,'HEC':14,'HHE':15,'HHC':16,'HCE':17,'HCH':18,'HCC':19,
        #                   'CEE':20,'CEH':21,'CEC':22,'CHE':23,'CHH':24,'CHC':25,'CCE':26,'CCH':27} #use number 28 to represent unknown 3-mers
        
        self.max_seq_len=maxlen
        self.maxlen=maxlen
        self.BioVec_name_dict=BioVec_name_dict
        self.BioVec_name_keys=self.BioVec_name_dict.keys()
        self.prot_names=[]
        self.seqs=[]
        self.seqlen=[]
        #self.PseAAC_mat=[]
        self.seq_mats=[]
        # self.ss_mats=[]
        # self.ss_sparse_mats=[]
        # self.ss_sparse_mats2=[] #Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding aa3mer
        # self.aa_ss_mixed_mats=[]
        self.prot_3mer_mats=[]
        # self.ss_3mer_mats=[]
        #self.seq_train_mats=[]
        #self.seq_val_mats=[]
        self.labels=[]
        self.train_labels=[]
        self.val_labels=[]
        self.usable_files=[]
        self.batch_id=0
        #max_seq=0
        for seq_file, class_label in zip(files, class_labels):
            try:
                seq, seq_len = self.get_sequence(seq_file)
                self.seqs.append(seq)
                self.seqlen.append(seq_len)
                self.labels.append(class_label)
                self.usable_files.append(seq_file)
                if seq_len>self.max_seq_len:
                    self.max_seq_len=seq_len
                    #max_seq=seq_file
            except ValueError:
                print("Value Error: errors in the sequence loading of {}.".format(seq_file))

        print("The maximum length of the sequences is : "+str(self.max_seq_len))

        #self.padding_maxlen=int((self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+self.maxlen/2)
        if self.max_seq_len > self.maxlen:
            self.padding_maxlen=int((self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+self.maxlen/2)
        else:
            self.padding_maxlen=self.maxlen+2
        
        for seq in self.seqs:
            seq_mat, prot_3mer_mat = self.encode_protein(seq)
            self.seq_mats.append(seq_mat)
            self.prot_3mer_mats.append(prot_3mer_mat)

        
        if len(self.seqs)!=len(self.labels):
            raise ValueError('The length of generated seq vector and class_labels list should be same.')
    
        self.seq_mats=np.array(self.seq_mats)
        self.prot_3mer_mats=np.array(self.prot_3mer_mats)
        self.seqlen=np.array(self.seqlen)
        self.labels=np.array(self.labels)
        self.usable_files=np.array(self.usable_files)

        
    def get_sequence(self, seq_file):
        #print seq_file
        f=open(seq_file, 'r')
        s=f.read()
        try:
            s=''.join(s.strip('* \n').split('\n')[1:])
            s=s.replace('*','')
            s=s.replace('X','')
        except:
            print("invalid sequence file:"+seq_file)
            f.close()
            return None, None

        f.close()
        ## Check invalid letter in the sequence. If there is an invalid letter, the string behind the letter will be discarded.
        seq=[]
        for x in s:
            seq.append(x)
            if not(x in self.aa_code):
                break
    
        seq=''.join(seq)
        return seq, len(seq)
    
    def encode_amino_acid(self, aa): #convert the AA to a integer, rather than a binary vector.
        aa=aa.upper()
        v=self.aa_code[aa]
        return v
            
    def encode_secondary_structure(self, ss):
        ss=ss.upper()
        v=self.ss_code[ss]
        return v
    
    def encode_secondary_structure_sparse(self, ss):
        ss=ss.upper()
        v=self.ss_sparse_code[ss]
        return v
    
    def encode_aa_3mer(self, mer):
        mer=mer.upper()
        if mer in self.BioVec_name_keys:
            v=self.BioVec_name_dict[mer]
        else:
            v=9048 # use the index of '<unk>' from the BioVec weights matrix.
        return v
    
    def encode_protein(self, prot_seq):
        """
        return: protein_sequence_using_digits, secondary_structure_sequence_using_digits, mixed_aa_ss_using_digits, protein_3mer_sequence_using_digits
        """
        length=len(prot_seq)
        prot_seq_l=[self.encode_amino_acid(aa) for aa in prot_seq if aa in self.aa_code]

        prot_seq_l+=[0]*(self.padding_maxlen-length)        
        prot_3mer_seq_l=[]
        for i in range(len(prot_seq)-2):
            prot_3mer_seq_l.append(self.encode_aa_3mer(prot_seq[i:i+3]))
        
        #prot_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        prot_3mer_seq_l+=[0]*(self.padding_maxlen-(length))
        
        #return csr_matrix(np.array(seq_l))
        return np.array(prot_seq_l), np.array(prot_3mer_seq_l)
        
    def get_aa3mer_mats(self):
        return self.prot_3mer_mats
    
    def get_class_labels(self):
        return self.labels

    def get_seqlens(self):
        return self.seqlen

    def get_usable_files(self):
        return self.usable_files

    def get_maxlen(self):
        return self.max_seq_len

class Protein_Sequence_Input5_2:
    """
    The version for sequence inputs rather than seqfiles input.
    """
    def __init__(self, seq_name, prot_seq, ss_seq, class_labels, BioVec_name_dict, maxlen=1500):
        """
        seq_name: list of strings. 
        prot_seq: list of strings. Protein's amino acid sequence.
        ss_seq: list of strings. Protein's amino acid sequence.
        class_labels: a list recording the RBP identity (True or False) of each protein mentioned in "files", best in numpy.array format.
        BioVec_name_dict: dictionary, a dictory telling the mapping between the aa triplet (e.g, HKE) and their index in this analysis.
        max_seqlen: int, the maximum length of amino acids that can be retained.
        """
        if len(prot_seq)!=len(ss_seq) :
            raise ValueError("The length of amino acid sequences' and secondary structure sequences' lists should be same.")
        
        if len(prot_seq)!=len(class_labels) :
            raise ValueError("The length of sequences' and class labels' lists should be same.")

        self.aa_code={'H':1, 'K':2, 'D':3, 'E':4,
                 'S':5, 'T':6, 'N':7, 'Q':8, 'C':9,
                 'U':10, 'G':11, 'P':12, 'A':13, 'V':14,
                 'I':15, 'L':16, 'M':17, 'F':18, 'Y':19,
                 'W':20, 'R':21}
        self.ss_code={'E':1, 'H':2, 'C':3}
        self.ss_sparse_code={'E':[1,0,0], 'H':[0,1,0], 'C':[0,0,1]}
        self.ss_mer_code={'EEE':1, 'HHH':2, 'CCC':3,
                          'EEH':4, 'EEC':5,'EHE':6,'EHH':7,'EHC':8,'ECE':9,'ECH':10,'ECC':11,
                          'HEE':12,'HEH':13,'HEC':14,'HHE':15,'HHC':16,'HCE':17,'HCH':18,'HCC':19,
                          'CEE':20,'CEH':21,'CEC':22,'CHE':23,'CHH':24,'CHC':25,'CCE':26,'CCH':27} #use number 28 to represent unknown 3-mers
        self.max_seq_len=maxlen
        self.maxlen=maxlen
        self.BioVec_name_dict=BioVec_name_dict
        self.BioVec_name_keys=self.BioVec_name_dict.keys()
        self.prot_names=[]
        self.seqs=[]
        self.seqlen=[]
        #self.PseAAC_mat=[]
        self.seq_mats=[]
        self.ss_mats=[]
        self.ss_sparse_mats=[]
        self.ss_sparse_mats2=[] #Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding aa3mer
        self.aa_ss_mixed_mats=[]
        self.prot_3mer_mats=[]
        self.ss_3mer_mats=[]
        #self.seq_train_mats=[]
        #self.seq_val_mats=[]
        self.labels=[]
        self.usable_seqs=[]
        self.batch_id=0
        #max_seq=0
        for seq_name, seq, ss_seq, class_label in zip(seq_name, prot_seq, ss_seq, class_labels):
            seq_len = len(seq)
            self.seqs.append((seq, ss_seq))
            self.seqlen.append(seq_len)
            self.labels.append(class_label)
            self.usable_seqs.append(seq_name)
            if seq_len>self.max_seq_len:
                self.max_seq_len=seq_len
                    #max_seq=seq_file
            
        print("The maximum length of the sequences is : "+str(self.max_seq_len))
        self.padding_maxlen=int((self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+self.maxlen/2)

        for seq, ss_seq in self.seqs:
            seq_mat, ss_mat, aa_ss_mixed_mat, prot_3mer_mat, ss_3mer_mat, ss_sparse_mat, ss_sparse_mat2 = self.encode_protein(seq, ss_seq)
            self.seq_mats.append(seq_mat)
            self.ss_mats.append(ss_mat)
            self.aa_ss_mixed_mats.append(aa_ss_mixed_mat)
            self.prot_3mer_mats.append(prot_3mer_mat)
            self.ss_3mer_mats.append(ss_3mer_mat)
            self.ss_sparse_mats.append(ss_sparse_mat)
            self.ss_sparse_mats2.append(ss_sparse_mat2)
                
        #TODO: Split into training, validation and test dataset
        self.seq_mats=np.array(self.seq_mats)
        self.ss_mats=np.array(self.ss_mats)
        self.ss_sparse_mats=np.array(self.ss_sparse_mats)
        self.ss_sparse_mats2=np.array(self.ss_sparse_mats2)
        self.aa_ss_mixed_mats=np.array(self.aa_ss_mixed_mats)
        self.prot_3mer_mats=np.array(self.prot_3mer_mats)
        self.ss_3mer_mats=np.array(self.ss_3mer_mats)
        #self.PseAAC_mat=np.array(self.PseAAC_mat)
        self.seqlen=np.array(self.seqlen)
        self.labels=np.array(self.labels)
        self.usable_seqs=np.array(self.usable_seqs)
    
    def encode_amino_acid(self, aa): #convert the AA to a integer, rather than a binary vector.
        aa=aa.upper()
        v=self.aa_code[aa]
        return v

    def encode_secondary_structure(self, ss):
        ss=ss.upper()
        v=self.ss_code[ss]
        return v
    
    def encode_secondary_structure_sparse(self, ss):
        ss=ss.upper()
        v=self.ss_sparse_code[ss]
        return v
    
    def encode_aa_3mer(self, mer):
        mer=mer.upper()
        if mer in self.BioVec_name_keys:
            v=self.BioVec_name_dict[mer]
        else:
            v=9048 # use the index of '<unk>' from the BioVec weights matrix.
        return v
    
    def encode_ss_3mer(self, mer):
        mer=mer.upper()
        if mer in self.ss_mer_code:
            v=self.ss_mer_code[mer]
        else:
            v=28
        return v
    
    def encode_protein(self, prot_seq, ss_seq):
        """
        return: protein_sequence_using_digits, secondary_structure_sequence_using_digits, mixed_aa_ss_using_digits, protein_3mer_sequence_using_digits
        """
        length=len(prot_seq)
        # prot_seq_l=[self.encode_amino_acid(aa) for aa in prot_seq]
        # prot_seq_l+=[0]*(self.max_seq_len-length)
        # ss_seq_l=[self.encode_secondary_structure(ss) for ss in ss_seq]
        # ss_seq_l+=[0]*(self.max_seq_len-length)
        # ss_seq_sparse_l=[self.encode_secondary_structure_sparse(ss) for ss in ss_seq]
        # ss_seq_sparse2_l=ss_seq_sparse_l[1:-1]
        # ss_seq_sparse_l+=[[0,0,0]]*(self.max_seq_len-length)
        # ss_seq_sparse2_l+=[[0,0,0]]*(self.max_seq_len-length)

        # prot_3mer_seq_l=[]
        # for i in range(len(prot_seq)-2):
        #     prot_3mer_seq_l.append(self.encode_aa_3mer(prot_seq[i:i+3]))
        
        # #prot_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        # prot_3mer_seq_l+=[0]*(self.max_seq_len-(length))
        
        # ss_3mer_seq_l=[]
        # for i in range(len(ss_seq)-2):
        #     ss_3mer_seq_l.append(self.encode_ss_3mer(ss_seq[i:i+3]))
        
        # #ss_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        # ss_3mer_seq_l+=[0]*(self.max_seq_len-(length))

        #return csr_matrix(np.array(seq_l))
        prot_seq_l=[self.encode_amino_acid(aa) for aa in prot_seq]
        prot_seq_l+=[0]*(self.padding_maxlen-length)
        ss_seq_l=[self.encode_secondary_structure(ss) for ss in ss_seq]
        ss_seq_l+=[0]*(self.padding_maxlen-length)
        ss_seq_sparse_l=[self.encode_secondary_structure_sparse(ss) for ss in ss_seq]
        ss_seq_sparse2_l=ss_seq_sparse_l[1:-1]
        ss_seq_sparse_l+=[[0,0,0]]*(self.padding_maxlen-length)
        ss_seq_sparse2_l+=[[0,0,0]]*(self.padding_maxlen-length)

        prot_3mer_seq_l=[]
        for i in range(len(prot_seq)-2):
            prot_3mer_seq_l.append(self.encode_aa_3mer(prot_seq[i:i+3]))
        
        prot_3mer_seq_l+=[0]*(self.padding_maxlen-(length))
        
        ss_3mer_seq_l=[]
        for i in range(len(ss_seq)-2):
            ss_3mer_seq_l.append(self.encode_ss_3mer(ss_seq[i:i+3]))
        
        ss_3mer_seq_l+=[0]*(self.padding_maxlen-(length))

        return np.array(prot_seq_l), np.array(ss_seq_l), np.array(zip(prot_seq_l, np.array(ss_seq_l)+21)).flatten(), np.array(prot_3mer_seq_l), np.array(ss_3mer_seq_l), np.array(ss_seq_sparse_l), np.array(ss_seq_sparse2_l)
        
    
    def get_aa3mer_mats(self):
        return self.prot_3mer_mats

    def get_ss3mer_mats(self):
        return self.ss_3mer_mats
    
    def get_ss_sparse_mats(self):
        return self.ss_sparse_mats

    def get_ss_sparse_mats2(self): #Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding aa3mer
        return self.ss_sparse_mats2
    
    def get_class_labels(self):
        return self.labels

    def get_seqlens(self):
        return self.seqlen

    def get_usable_files(self):
        return self.usable_files

    def get_maxlen(self):
        return self.max_seq_len


class Protein_Sequence_Input5_2_noSS:
    """
    The version for sequence inputs rather than seqfiles input.
    """
    def __init__(self, seq_name, prot_seq, class_labels, BioVec_name_dict, maxlen=1500):
        """
        seq_name: list of strings, the names of each sequence. 
        prot_seq: list of strings. Protein's amino acid sequence.
        ss_seq: list of strings. Protein's amino acid sequence.
        class_labels: a list recording the RBP identity (True or False) of each protein mentioned in "files", best in numpy.array format.
        BioVec_name_dict: dictionary, a dictory telling the mapping between the aa triplet (e.g, HKE) and their index in this analysis.
        max_seqlen: int, the maximum length of amino acids that can be retained.
        """        
        if len(prot_seq)!=len(class_labels) :
            raise ValueError("The length of sequences' and class labels' lists should be same.")

        self.aa_code={'H':1, 'K':2, 'D':3, 'E':4,
                 'S':5, 'T':6, 'N':7, 'Q':8, 'C':9,
                 'U':10, 'G':11, 'P':12, 'A':13, 'V':14,
                 'I':15, 'L':16, 'M':17, 'F':18, 'Y':19,
                 'W':20, 'R':21}
        self.max_seq_len=maxlen
        self.maxlen=maxlen
        self.BioVec_name_dict=BioVec_name_dict
        self.BioVec_name_keys=self.BioVec_name_dict.keys()
        self.prot_names=[]
        self.seqs=[]
        self.seqlen=[]
        #self.PseAAC_mat=[]
        self.seq_mats=[]
        self.prot_3mer_mats=[]
        #self.seq_train_mats=[]
        #self.seq_val_mats=[]
        self.labels=[]
        self.usable_seqs=[]
        self.batch_id=0
        #max_seq=0
        for seq_name, seq, class_label in zip(seq_name, prot_seq, class_labels):
            seq_len = len(seq)
            self.seqs.append(seq)
            self.seqlen.append(seq_len)
            self.labels.append(class_label)
            self.usable_seqs.append(seq_name)
            if seq_len>self.max_seq_len:
                self.max_seq_len=seq_len
                    #max_seq=seq_file
            
        print("The maximum length of the sequences is : "+str(self.max_seq_len))
        self.padding_maxlen=int((self.max_seq_len/(self.maxlen/2))*(self.maxlen/2)+self.maxlen/2)

        for seq in self.seqs:
            seq_mat, prot_3mer_mat = self.encode_protein(seq)
            self.seq_mats.append(seq_mat)
            self.prot_3mer_mats.append(prot_3mer_mat)
        
        #TODO: Split into training, validation and test dataset
        self.seq_mats=np.array(self.seq_mats)
        self.prot_3mer_mats=np.array(self.prot_3mer_mats)
        #self.PseAAC_mat=np.array(self.PseAAC_mat)
        self.seqlen=np.array(self.seqlen)
        self.labels=np.array(self.labels)
        self.usable_seqs=np.array(self.usable_seqs)
    
    def encode_amino_acid(self, aa): #convert the AA to a integer, rather than a binary vector.
        aa=aa.upper()
        v=self.aa_code[aa]
        return v
    
    def encode_aa_3mer(self, mer):
        mer=mer.upper()
        if mer in self.BioVec_name_keys:
            v=self.BioVec_name_dict[mer]
        else:
            v=9048 # use the index of '<unk>' from the BioVec weights matrix.
        return v
    
    def encode_protein(self, prot_seq):
        """
        return: protein_sequence_using_digits, secondary_structure_sequence_using_digits, mixed_aa_ss_using_digits, protein_3mer_sequence_using_digits
        """
        # length=len(prot_seq)
        # prot_seq_l=[self.encode_amino_acid(aa) for aa in prot_seq]
        # prot_seq_l+=[0]*(self.max_seq_len-length)        
        # prot_3mer_seq_l=[]
        # for i in range(len(prot_seq)-2):
        #     prot_3mer_seq_l.append(self.encode_aa_3mer(prot_seq[i:i+3]))
        
        # #prot_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        # prot_3mer_seq_l+=[0]*(self.max_seq_len-(length))
        
        length=len(prot_seq)
        prot_seq_l=[self.encode_amino_acid(aa) for aa in prot_seq]
        prot_seq_l+=[0]*(self.padding_maxlen-length)        
        prot_3mer_seq_l=[]
        for i in range(len(prot_seq)-2):
            prot_3mer_seq_l.append(self.encode_aa_3mer(prot_seq[i:i+3]))
        
        #prot_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        prot_3mer_seq_l+=[0]*(self.padding_maxlen-(length))

        return np.array(prot_seq_l), np.array(prot_3mer_seq_l)
    
    def get_aa3mer_mats(self):
        return self.prot_3mer_mats
    
    def get_class_labels(self):
        return self.labels

    def get_seqlens(self):
        return self.seqlen

    def get_usable_files(self):
        return self.usable_files

    def get_maxlen(self):
        return self.max_seq_len


class Protein_Sequence_Input5_bk:
    """
    Different from version Protein_Sequence_Input4: Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding amino acid (from aa3mer)
    """
    def __init__(self, files, class_labels, BioVec_name_dict, max_seqlen=1500):
        """
        files: a list of sequence filenames including the absolute path, best in numpy.array format.
        class_labels: a list recording the RBP identity (True or False) of each protein mentioned in "files", best in numpy.array format.
        PPI_feature_vectors: a list of PPI feature vectors for each protein mentioned in "files", best in numpy.array format.
        BioVec_name_dict: dictionary, a dictory telling the mapping between the aa triplet (e.g, HKE) and their index in this analysis.
        max_seqlen: int, the maximum length of amino acids that can be retained.
        """
        if len(files)!=len(class_labels) :
            raise ValueError('The length of files list and class_labels list should be same.')
            
        self.aa_code={'H':1, 'K':2, 'D':3, 'E':4,
                 'S':5, 'T':6, 'N':7, 'Q':8, 'C':9,
                 'U':10, 'G':11, 'P':12, 'A':13, 'V':14,
                 'I':15, 'L':16, 'M':17, 'F':18, 'Y':19,
                 'W':20, 'R':21}
        self.ss_code={'E':1, 'H':2, 'C':3}
        self.ss_sparse_code={'E':[1,0,0], 'H':[0,1,0], 'C':[0,0,1]}
        self.ss_mer_code={'EEE':1, 'HHH':2, 'CCC':3,
                          'EEH':4, 'EEC':5,'EHE':6,'EHH':7,'EHC':8,'ECE':9,'ECH':10,'ECC':11,
                          'HEE':12,'HEH':13,'HEC':14,'HHE':15,'HHC':16,'HCE':17,'HCH':18,'HCC':19,
                          'CEE':20,'CEH':21,'CEC':22,'CHE':23,'CHH':24,'CHC':25,'CCE':26,'CCH':27} #use number 28 to represent unknown 3-mers
        self.max_seq_len=max_seqlen
        self.BioVec_name_dict=BioVec_name_dict
        self.BioVec_name_keys=self.BioVec_name_dict.keys()
        self.prot_names=[]
        self.seqs=[]
        self.seqlen=[]
        #self.PseAAC_mat=[]
        self.seq_mats=[]
        self.ss_mats=[]
        self.ss_sparse_mats=[]
        self.ss_sparse_mats2=[] #Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding aa3mer
        self.aa_ss_mixed_mats=[]
        self.prot_3mer_mats=[]
        self.ss_3mer_mats=[]
        #self.seq_train_mats=[]
        #self.seq_val_mats=[]
        self.labels=[]
        self.train_labels=[]
        self.val_labels=[]
        self.usable_files=[]
        self.batch_id=0
        #max_seq=0
        for seq_file, class_label in zip(files, class_labels):
            try:
                seq, ss_seq, seq_len = self.get_sequence(seq_file)
                self.seqs.append((seq, ss_seq))
                self.seqlen.append(seq_len)
                self.labels.append(class_label)
                self.usable_files.append(seq_file)
                if seq_len>self.max_seq_len:
                    self.max_seq_len=seq_len
                    #max_seq=seq_file

            except ValueError:
                print(seq_file+" is ignored, because of the length conflict in seq_file and ss_seq_file.")
        
        #print max_seq
        
        if len(self.seqs)!=len(self.labels):
            raise ValueError('The length of generated seq vector and class_labels list should be same.')
            
        print("The maximum length of the sequences is : "+str(self.max_seq_len))
        
        for seq, ss_seq in self.seqs:
            seq_mat, ss_mat, aa_ss_mixed_mat, prot_3mer_mat, ss_3mer_mat, ss_sparse_mat, ss_sparse_mat2 = self.encode_protein(seq, ss_seq)
            self.seq_mats.append(seq_mat)
            self.ss_mats.append(ss_mat)
            self.aa_ss_mixed_mats.append(aa_ss_mixed_mat)
            self.prot_3mer_mats.append(prot_3mer_mat)
            self.ss_3mer_mats.append(ss_3mer_mat)
            self.ss_sparse_mats.append(ss_sparse_mat)
            self.ss_sparse_mats2.append(ss_sparse_mat2)
            
        #TODO: Split into training, validation and test dataset
        self.seq_mats=np.array(self.seq_mats)
        self.ss_mats=np.array(self.ss_mats)
        self.ss_sparse_mats=np.array(self.ss_sparse_mats)
        self.ss_sparse_mats2=np.array(self.ss_sparse_mats2)
        self.aa_ss_mixed_mats=np.array(self.aa_ss_mixed_mats)
        self.prot_3mer_mats=np.array(self.prot_3mer_mats)
        self.ss_3mer_mats=np.array(self.ss_3mer_mats)
        #self.PseAAC_mat=np.array(self.PseAAC_mat)
        self.seqlen=np.array(self.seqlen)
        self.labels=np.array(self.labels)
        self.usable_files=np.array(self.usable_files)

        
    def get_sequence(self, seq_file):
        #print seq_file
        f=open(seq_file, 'r')
        s=f.read()
        try:
            s=''.join(s.strip('* \n').split('\n')[1:])
            s=s.replace('*','')
            s=s.replace('X','')
        except:
            print("invalid sequence file:"+seq_file)
            f.close()
            return None, None
            
        f.close()
        seq_file2=seq_file.replace('.fasta', '.spd3')
        tmp=pd.read_table(seq_file2)
        try:
            tmp=tmp[tmp['AA']!='*']
            tmp=tmp[tmp['AA']!='X']
            ss=''.join(list(tmp['SS']))
            if len(s)!=len(ss):
                #print len(s), len(ss)
                raise ValueError('The length of sequence and SS sequence is different. Files: '+seq_file+', '+seq_file2+'.')
        except:
            print('Problem with {}'.format(seq_file2))
            raise ValueError('Problem with {}'.format(seq_file2))

        return s, ss, len(s)
    
    def encode_amino_acid(self, aa): #convert the AA to a integer, rather than a binary vector.
        aa=aa.upper()
        v=self.aa_code[aa]
        return v

    def encode_secondary_structure(self, ss):
        ss=ss.upper()
        v=self.ss_code[ss]
        return v
    
    def encode_secondary_structure_sparse(self, ss):
        ss=ss.upper()
        v=self.ss_sparse_code[ss]
        return v
    
    def encode_aa_3mer(self, mer):
        mer=mer.upper()
        if mer in self.BioVec_name_keys:
            v=self.BioVec_name_dict[mer]
        else:
            v=9048 # use the index of '<unk>' from the BioVec weights matrix.
        return v
    
    def encode_ss_3mer(self, mer):
        mer=mer.upper()
        if mer in self.ss_mer_code:
            v=self.ss_mer_code[mer]
        else:
            v=28
        return v
    
    def encode_protein(self, prot_seq, ss_seq):
        """
        return: protein_sequence_using_digits, secondary_structure_sequence_using_digits, mixed_aa_ss_using_digits, protein_3mer_sequence_using_digits
        """
        length=len(prot_seq)
        prot_seq_l=[self.encode_amino_acid(aa) for aa in prot_seq]
        prot_seq_l+=[0]*(self.max_seq_len-length)
        ss_seq_l=[self.encode_secondary_structure(ss) for ss in ss_seq]
        ss_seq_l+=[0]*(self.max_seq_len-length)
        ss_seq_sparse_l=[self.encode_secondary_structure_sparse(ss) for ss in ss_seq]
        ss_seq_sparse2_l=ss_seq_sparse_l[1:-1]
        ss_seq_sparse_l+=[[0,0,0]]*(self.max_seq_len-length)
        ss_seq_sparse2_l+=[[0,0,0]]*(self.max_seq_len-length)
        
        prot_3mer_seq_l=[]
        for i in range(len(prot_seq)-2):
            prot_3mer_seq_l.append(self.encode_aa_3mer(prot_seq[i:i+3]))
        
        #prot_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        prot_3mer_seq_l+=[0]*(self.max_seq_len-(length))
        
        ss_3mer_seq_l=[]
        for i in range(len(ss_seq)-2):
            ss_3mer_seq_l.append(self.encode_ss_3mer(ss_seq[i:i+3]))
        
        #ss_3mer_seq_l+=[0]*(self.max_seq_len-(length-2))
        ss_3mer_seq_l+=[0]*(self.max_seq_len-(length))


        #return csr_matrix(np.array(seq_l))
        return np.array(prot_seq_l), np.array(ss_seq_l), np.array(zip(prot_seq_l, np.array(ss_seq_l)+21)).flatten(), np.array(prot_3mer_seq_l), np.array(ss_3mer_seq_l), np.array(ss_seq_sparse_l), np.array(ss_seq_sparse2_l)
        
    
    def get_aa3mer_mats(self):
        return self.prot_3mer_mats

    def get_ss3mer_mats(self):
        return self.ss_3mer_mats
    
    def get_ss_sparse_mats(self):
        return self.ss_sparse_mats

    def get_ss_sparse_mats2(self): #Shift ss_seq (i.e. remove the first and end of the sequence) in order to match the position of corresponding aa3mer
        return self.ss_sparse_mats2
    
    def get_class_labels(self):
        return self.labels

    def get_seqlens(self):
        return self.seqlen

    def get_usable_files(self):
        return self.usable_files
