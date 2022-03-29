CUDA_VISIBLE_DEVICES=$5
model_topic=$4
seed=42
topic=$6

aug_type=$1
multiple_number=$2
loss_type=$3
label_method=$7

analysis_data_dir="./log"
log_dir="./log"

echo $model_topic $topic $label_method $aug_type $multiple_number

for metrics in tau
do
    python ./main.py --seed=$seed --model_dataset=$model_topic --log_dir=$log_dir --task=label_score_analysis --analysis_encode_method=term --analysis_distance_method=Cosine --analysis_cluster_method=AgglomerativeClustering --analysis_data_dir=$analysis_data_dir --analysis_data_type=dataset --analysis_compare_method=correlation --analysis_correlation_method=$metrics --graph_distance=co_occurance --graph_kernel=cluster --loss_type=$loss_type --augment_type=$aug_type --multiple_number=$multiple_number --label_method=$label_method --dataset=$topic
done