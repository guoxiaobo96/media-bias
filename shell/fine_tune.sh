topic=$4
dataset=$5
seed=42
batch_size=16

aug_type=$1
multiple_number=$2
loss_type=$3

echo $topic $aug_type $multiple_number $loss_type
output_dir="./model/model_$topic/$seed"
data_dir="./data/data_$topic"
log_dir="./log"

echo "$dataset"
python ./main.py  --task=train_lm --output_dir=$output_dir/$dataset --data_dir=$data_dir --model_type=bert --model_name_or_path=bert-base-uncased --do_train --do_eval --loss_type=$loss_type --line_by_line --overwrite_output_dir --dataset=$dataset  --augment_type=$aug_type --multiple_number=$multiple_number  --per_device_train_batch_size=$batch_size --num_train_epochs=10 --log_level=error --seed=$seed --model_dataset=$topic --log_dir=$log_dir --load_best_model_at_end --evaluation_strategy=epoch --model_dataset=$topic --logging_steps=0

rm -rf $output_dir/$dataset/$loss_type/$aug_type/$multiple_number/checkpoint-*