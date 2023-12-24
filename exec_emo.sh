LR=2e-5
learning_rate_pretrained=2e-6
GPU=1
lay=1
epochs=50
diff=bert_45
# dataset=MELD_three #IEMOCAP 
pretrain_dir=snapshot_bert2
# 1>$log_path$diff 2>&1  -load_model \-Robert_hubert_pre \-no_context \
echo ========= lr=$LR ==============
for dataset in MELD_three #IEMOCAP
do
echo ----  $dataset--$LR--$diff-----
log_path=EmotionCheckpoint/$dataset/test_true.log
nohup ~/anaconda3/envs/t17py36/bin/python3 -u EmoMain.py \
-lr $LR \
-gpu $GPU \
-use_bert \
-epochs $epochs \
-learning_rate_pretrained $learning_rate_pretrained \
-pretrain_dir $pretrain_dir \
-dataset $dataset \
-diff $diff \
-load_model \
-used_modalities all \
1>$log_path$diff 2>&1
done