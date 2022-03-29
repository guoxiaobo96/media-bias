aug_type=$1
multiple_number=$2


topic=$3


echo $topic $aug_type $multiple_number
data_dir="./data/data_$topic"

python ./main.py --data_dir=$data_dir --task=data_collect --dataset=$topic --augment_type=$aug_type --multiple_number=$multiple_number --seed=42  --model_dataset=$topic
