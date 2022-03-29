source ./env/bin/activate
topic=$1
for dataset in Breitbart CBS CNN Fox HuffPost NPR NYtimes usatoday wallstreet washington
do
    ./shell/fine_tune.sh no_augmentation 1 mlm $topic $dataset
done
