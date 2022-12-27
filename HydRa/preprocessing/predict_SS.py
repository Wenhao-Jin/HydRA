#!/home/wjin/envs/wjin/bin/python
import os
import pandas as pd

##SPIDER2
def predict_2ary_structure_spider2(prot_file, out_dir):
    if(not os.path.exists(prot_file.replace('.fasta', '.spd3'))):
        os.chdir(out_dir)
        try:
            os.system('/home/wjin/software/SPIDER2_local/misc/run_local2.sh '+prot_file)
        except:
            print(prot_file.split('/')[-1].split('.')[0]+" hasn't been processed.")

def extract_SS(prot_spd3_file):
    tmp=pd.read_table(prot_spd3_file)
    tmp=tmp[tmp['AA']!='*']
    tmp=tmp[tmp['AA']!='X']
    return ''.join(list(tmp['SS'])).strip(' ')