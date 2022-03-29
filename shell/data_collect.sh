echo "Generate data from the raw data"
for topic in climate-change corporate-tax drug-policy gay-marriage obamacare
do
    ./shell/data_aug.sh duplicate 1 $topic
    for dataset in Breitbart CBS CNN Fox HuffPost NPR NYtimes usatoday wallstreet washington
    do
        mkdir ./data/data_$topic/42/$dataset/no_augmentation
        mv ./data/data_$topic/42/$dataset/duplicate/1 ./data/data_$topic/42/$dataset/no_augmentation
        rm -rf ./data/data_$topic/42/$dataset/duplicate
    done
done


echo "Lable masked token with bigram method"
for data_topic in climate-change corporate-tax drug-policy gay-marriage obamacare
do
    echo "$data_topic"
    python ./main.py --seed=42 --task=label_masked_token --data_dir=./data/data_$data_topic --model_type=bert --model_name_or_path=bert-base-uncased --loss_type=class --dataset=all --augment_type=original --label_method=bigram_outer --model_dataset=$data_topic
done
