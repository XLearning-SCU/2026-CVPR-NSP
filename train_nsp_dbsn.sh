#!/bin/bash

# ===========================================
BS=16
LR=1e-4
SEED=10
NG=1
# ===========================================
experiment_home_dir="exp_nsp_dbsn"

experiment_name=$(python name_dbsn.py --bs $BS --seed $SEED --lr $LR --ng $NG)

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

cp "dataloader.py" "${experiment_dir}"
cp "DBSNl.py" "${experiment_dir}"
cp "train_nsp_dbsn.py" "${experiment_dir}"
cp "utils.py" "${experiment_dir}"
cp "train_nsp_dbsn.sh" "${experiment_dir}"

cd ${experiment_dir}
export PYTHONPATH=$PWD:$PYTHONPATH
python train_nsp_dbsn.py \
  --train_dir "../../datasets/SIDD/SIDD_Medium_Srgb/Data/*/*_NOISY_*.PNG" \
  --val_rn_dir "../../datasets/SIDD/valid_data/sidd_v_noisy/*.png" \
  --val_gt_dir "../../datasets/SIDD/valid_data/sidd_v_clean/*.png" \
  --ng $NG --bs $BS --gpuid "0" --lr $LR --seed $SEED \
  # --pretrained_model "../b16-sd10-1.0e-04-ng1-/ckpts/_ckpt_it25_3421_8433.pth"
