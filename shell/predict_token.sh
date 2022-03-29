source ./env/bin/activate
topic=$1
for dataset in Breitbart CBS CNN Fox HuffPost NPR NYtimes usatoday wallstreet washington
do
    log_dir="./log"
    for target_topic in climate-change corporate-tax drug-policy gay-marriage obamacare
    do
        ./shell/label_score_predict.sh no_augmentation 1 mlm $topic $dataset $target_topic $log_dir
    done
done
