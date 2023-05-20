#!/bin/bash

#bash /media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/EVE/train_s2.sh
if [ ! -n "$1" ]; then
  echo "please input a EXP_ID!"
  exit 8
else
  EXP_ID=$1
  echo "EXP_ID is $EXP_ID"
fi

cd /media/titan1/52f525b8-0ae0-4e4c-8c95-8c8641e2743a/ldz_proj/EVE
source /home/titan1/anaconda3/bin/activate EVE
export CUDA_VISIBLE_DEVICES=2,3,4,5
which python
which torchrun

torchrun --nproc_per_node=4 sam_scripts/train_EVE.py \
  --exp_id $EXP_ID \
  --stage 2 \
  --s2_batch_size 4 \
  --s2_num_frames 4 \
  --log_text_interval 100 \
  --model_type vit_h \
#  --s2_iterations 60000 \
#  --s2_steps 65000 \
#  --save_network_interval 3000 \
#  --save_checkpoint_interval 6000 \
#  --load_checkpoint saves/debug_s2/debug_s2_checkpoint_48000.pth

#| J&F-Mean | J-Mean | J-Recall | J-Decay | F-Mean  | F-Recall | F-Decay
#| 0.831953 | 0.805476 | 0.893966 | 0.071745 | 0.85843 | 0.929103 | 0.098502
