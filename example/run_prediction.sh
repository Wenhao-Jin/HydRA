#!/usr/bin/bash

proteinBERT_RBP_file="/tscc/nfs/home/wjin/projects/RBP_pred/RBP_identification/HydRa2.0/data/ProteinBERT/ProteinBERT_TrainWithWholeProteinSet_defaultSetting_ModelFile.pkl"

# Run HydRA prediction
mkdir ./results

### You can choose either of the following options according to your needs.
## Option1: Run HydRA with only protein sequences as input.
HydRa2_predict --seq_dir ./seqs --proteinBERT_modelfile $proteinBERT_RBP_file --outdir ./results -n test_pred_seqOnly --no-PIA --no-PPA

## Option2: Run HydRA with protein sequences and protein interaction networks (HydRA default PPI networks) as input.
HydRa2_predict --seq_dir ./seqs --proteinBERT_modelfile $proteinBERT_RBP_file --outdir ./results -n test_pred_seqNppi

# Generate occlusion maps
occlusion_map3 -s ./seqs \
--out_dir ./results \
--proteinBERT_modelfile $proteinBERT_RBP_file \
-n test_pred \
--draw_ensemble_only

# To re-train the HydRA model, please follow the tutorial on our github page: https://github.com/YeoLab/HydRA OR https://github.com/Wenhao-Jin/HydRA 
