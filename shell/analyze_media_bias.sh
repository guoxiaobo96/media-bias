source ./env/bin/activate
clear

CUDA_VISIBLE_DEVICES=0


for method in no_augmentation
do
    for factor in 1
    do
        for topic in climate-change  corporate-tax  drug-policy  gay-marriage  obamacare
        do
            for target_topic in climate-change  corporate-tax  drug-policy  gay-marriage  obamacare
            do
                ./shell/label_score_analysis.sh $method $factor mlm $topic $CUDA_VISIBLE_DEVICES $target_topic bigram_outer
            done
        done
    done
done