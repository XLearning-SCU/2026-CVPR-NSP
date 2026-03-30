#!/bin/bash
# ===========================================
BS=4
PS=640
LR=3e-4
SEED=10
CHOSEN=1
# ===========================================
experiment_home_dir="exp_nsp_tbsn"

experiment_name=$(python name_tbsn.py --bs $BS --ps $PS --seed $SEED --lr $LR --chosen $CHOSEN)

experiment_dir="${experiment_home_dir}/${experiment_name}"
echo "experiment dir: ${experiment_dir}"

if [ ! -d "${experiment_home_dir}" ]
then
  mkdir "${experiment_home_dir}"
fi

if [ ! -d "${experiment_dir}" ]
then
  mkdir "${experiment_dir}"
else
  echo "experiment dir exists"
fi

cp "dataloader_tbsn.py" "${experiment_dir}"
cp "TBSN.py" "${experiment_dir}"
cp "train_nsp_tbsn.py" "${experiment_dir}"
cp "utils.py" "${experiment_dir}"
cp "train_nsp_tbsn.sh" "${experiment_dir}"

cd ${experiment_dir}
export PYTHONPATH=$PWD:$PYTHONPATH
python train_nsp_tbsn.py \
  --train_dir "../../datasets/SIDD/SIDD_Medium_Srgb/Data/*/*_NOISY_*.PNG" \
  --val_rn_dir "../../datasets/SIDD/valid_data/sidd_v_noisy/*.png" \
  --val_gt_dir "../../datasets/SIDD/valid_data/sidd_v_clean/*.png" \
  --bs $BS --ps $PS --lr $LR --chosen $CHOSEN --seed $SEED --gpuid "0"
