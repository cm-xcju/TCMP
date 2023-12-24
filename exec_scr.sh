#!bin/bash
#  \ 1>$save_dir/pre_test.log 2>&1
LR=2e-4
GPU=1
save_dir=snapshot_bert_audio
echo ========= lr=$LR ==============
for iter in 1
do
echo --- $Enc - $Dec $iter ---
  ~/anaconda3/envs/t17py36/bin/python3 -u LMMain.py \
-lr $LR \
-gpu $GPU \
-use_bert \
-epochs 30 \
-save_dir $save_dir \
-layers 1 \
-patience 10 
done
