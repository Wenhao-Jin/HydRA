#!/usr/bin/env python
import pandas as pd
import numpy as np
import multiprocessing as mp
import os
from random import shuffle

##SPIDER2
def predict_2ary_structure_spider2(prot_file, out_dir):
    if(not os.path.exists(prot_file.replace('.fasta', '.spd3'))):
        os.chdir(out_dir)
        try:
            os.system('/home/wjin/software/SPIDER2_local/misc/run_local2.sh '+prot_file)
        except:
            print(prot_file.split('/')[-1].split('.')[0]+" hasn't been processed.")


if __name__=='__main__':
	out_dir='/home/wjin/data3/Coronavirus/processed'  ##for SPIDER2, one weird thing is the "out_dir" must be the same as "seq_dir"
	seq_dir=out_dir

	prot_files=set(filter(lambda x:x.endswith('.fasta'), os.listdir(out_dir)))


	pool1=mp.Pool(processes=None)
	results=[pool1.apply_async(predict_2ary_structure_spider2, args=(os.path.join(seq_dir, protfile), out_dir)) for protfile in prot_files if os.path.exists(os.path.join(seq_dir, protfile)) and not os.path.exists(os.path.join(seq_dir, protfile.replace('.fasta','.pssm')))]
	results=[p.get() for p in results]