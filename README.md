# Measuring Media Bias via Masked Language Modeling
This repo covers an reference implementation for the following papers:
Measuring Media Bias via Masked Language Modeling.
## Overview
In this work, we present a model for mesuaring media bias via masked language modeling. Our approach includes:
1. Fine-tune language model on the articles of each media outlet.
2. Automatically generate masked prompts for the task of masked language modeling.
3. Predict the masked token and calculate media bias based on the probability of masked token.

You can find more details of this work in our paper.

## Experiment results
In this repo, we inlcude the train and dev datasets of all five topics, the predicted tokens for the masked prompts, and the performance reported in the paper. These results can be used directly for further research.

**(1) Data**

In the fodler ```./data```, we include all data used for our experiments. You can find the data for each topic and each media outlet in the following path:
```
./data/data_$topic/42/$media_outlet/no_augmentation/1
```
where $topic should be one of the 5 topics in our paper, and $media_outlet should be one of the 10 media outlets in our paper.

The masked tokens for the chosen topic are kept in the file:
```
./data/data_$topic/42/all/original/1/en.masked.bigram_outer
```

The ground-truth data can be found in:
```
    ./data/ground-truth
```

**(2) Predicted Token**

We keep the predicted token for each topic and media outlets in the following folder:
```
    ./log/$model_topic/$target_topic/42/mlm/bigram_outer/no_augmentation/1/json
```
where $model_topic means the topic we fine-tune the language model, and $target_topic is the topic we predict on. For in-domain experiments, the $model_topic and $target_topic should be the same, and for out-of-domain experiments, the $model_topic and $target_topic are different.

For the item in the file, it will include the unmasked instance, the masked position, the predicted tokens and its probability.

**(3) Media bias performance**

We keep the performance of meida bias in the folder:
```
    ./analysis/$model_topic/$target_topic/42/mlm/bigram_outer/
```
where $model_topic means the topic we fine-tune the language model, and $target_topic is the topic we predict on. In the ```record``` file, we record the performance with different ground-truth.

We also keep the sentence level performance in the file:
```
    ./analysis/$model_topic/$target_topic/42/mlm/bigram_outer/no_augmentation/1/correlation/tau/term_sentence_$ground-truth.json
```
where $ground-truth means the ground-truth dataset.


## Running

In stead of using our results, you may aslo run the experiments by yourselves. You might use CUDA_VISIBLE_DEVICES to set proper number of GPUs. Before runing experiemnts, please back the following two folders: ```./analysis``` and  ```./log```


**(1) Requirements**

To run our code, please install all the dependency packages by using the following command:
```
pip install -r requirements.txt
chmod +x ./shell/*
```
**NOTE**: Different versions of packages (like pytorch, transformers, etc.) may lead to different results from the paper.

**(2) Generate ground truth data**

```
    python ./baseline.py
```
**(3) Train language model**

Our code is built on transformers and we use its 4.8.2 version. Other versions of transformers might cause unexpected errors.

To fine-tune the language model on each topic, you can run our example code below:
```
    topic=cliamge-change
    ./shell/tain_lm.sh $topic
```
where topic can be chosen from climate-change, corporate-tax, drug-policy, gay-marriage, obamacare

**(4) Predict token**

To predict the masked token on each topic, you can run our example code below:
```
    topic=cliamge-change
    ./shell/predict_token.sh $topic
```
where topic can be chosen from climate-change, corporate-tax, drug-policy, gay-marriage, obamacare. The predicted token will be in the ```./log``` folder.

**(5) Analye media bias**

To analyze the media bias based on the predicted tokens, you can run our example code below:
```
    ./shell/analyze_media_bias.sh
```
The generated analyze result will be shown in the ```./lanalysis``` folder