model_topic=$4
dataset=$5
seed=42
topic=$6

aug_type=$1
multiple_number=$2
loss_type=$3
label_method=bigram_outer

data_dir="./data/data_$topic"
model_dir="./model/model_$model_topic/$seed"
log_dir=$7

echo $dataset $topic $loss_type $aug_type $multiple_number

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main.py --task=label_score_predict --load_model --load_model_dir=$model_dir/$dataset --data_dir=$data_dir --model_type=bert --model_name_or_path=bert-base-cased --line_by_line --overwrite_output_dir --dataset=$dataset --loss_type=$loss_type --augment_type=$aug_type --multiple_number=$multiple_number --label_method=$label_method --seed=$seed  --model_dataset=$model_topic --log_dir=$log_dir --per_device_eval_batch_size=1
# rm -rf $model_dir/$dataset/$loss_type/$aug_type/$multiple_number/pytorch_model.bin