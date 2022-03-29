import torch
import numpy as np
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from copy import deepcopy
from random import sample

def random_label(original_sentence_list, sentence_percent, word_percent):
    masked_sentence_list = list()

    count_vectorize = CountVectorizer(min_df=10,ngram_range=(2,2),stop_words="english")
    tokenizer = count_vectorize.build_tokenizer()
    chosen_sentence_number = int(len(original_sentence_list)*sentence_percent)
    chosen_sentence_list = sample(original_sentence_list,chosen_sentence_number)

    for sentence in tqdm(chosen_sentence_list):
        if sentence['sentence'] == '':
            continue
        written_item = {"original_sentence":sentence["sentence"],"masked_sentence":list()}
        splited_sentence = tokenizer(sentence["sentence"])
        token_index_list = [i for i in range(len(splited_sentence))]
        chosen_index_list = sample(token_index_list,max(int(len(splited_sentence)*word_percent),1))
        for index in chosen_index_list:
            masked_sentence = splited_sentence[:index]+['[MASK]']+splited_sentence[index+1:]
            masked_sentence = ' '.join(masked_sentence)
            written_item["masked_sentence"].append(masked_sentence)
        if written_item["masked_sentence"] != []:
            masked_sentence_list.append(written_item)
    return masked_sentence_list

def ngram_outer_label(original_sentence_list, n_gram, min_df):
    masked_sentence_list = list()
    stemmer = PorterStemmer()
    count_vectorize = CountVectorizer(min_df=min_df,ngram_range=(n_gram,n_gram),stop_words="english")
    tokenizer = count_vectorize.build_tokenizer()
    raw_data = ['' for _ in range(10)]
    for item in tqdm(original_sentence_list,desc="load data"):
        sentence = ' '.join([stemmer.stem(word) for word in tokenizer(item['sentence'])])
        raw_data[item["label"]] = raw_data[item["label"]] + ' '+sentence
    count_vectorize.fit_transform(raw_data)
    for sentence in original_sentence_list:
        splited_sentence = tokenizer(sentence['sentence'])
    word_list = count_vectorize.get_feature_names()
    for sentence in tqdm(original_sentence_list,desc="mask token"):
        written_item = {"original_sentence":sentence["sentence"],"masked_sentence":list()}
        splited_sentence = tokenizer(sentence["sentence"])
        for index in range(len(splited_sentence)-(n_gram-1)):
            phrase = ''
            count = 0
            while len(phrase.split(' '))<n_gram:
                if count+index < len(splited_sentence):
                    stemmed_word = stemmer.stem(splited_sentence[count+index])
                    phrase = phrase + ' '+ stemmed_word
                    phrase = phrase.strip()
                    count += 1
                else:
                    break
            if len(phrase.split(' ')) < n_gram:
                break
            if phrase in word_list:
                full_phrase = splited_sentence[index:index+count]
                pre_masked_sentence = None
                post_masked_sentence = None
                if index != 0:
                    pre_masked_sentence = splited_sentence[:index-1]+['[MASK]']+splited_sentence[index:]
                    pre_masked_sentence = ' '.join(pre_masked_sentence)
                    written_item["masked_sentence"].append(pre_masked_sentence)
                if index + count <(len(splited_sentence)-1):
                    post_masked_sentence = splited_sentence[:index+count]+['[MASK]']+splited_sentence[index+count+1:]
                    post_masked_sentence = ' '.join(post_masked_sentence)
                    written_item["masked_sentence"].append(post_masked_sentence)
        if written_item["masked_sentence"] != []:
            masked_sentence_list.append(written_item)
    return masked_sentence_list

    masked_sentence_list = list()
    stemmer = PorterStemmer()
    count_vectorize = CountVectorizer(min_df=min_df,ngram_range=(n_gram,n_gram),stop_words="english")
    tokenizer = count_vectorize.build_tokenizer()
    raw_data = list()

    for item in original_sentence_list:
        sentence = ' '.join([stemmer.stem(word) for word in tokenizer(item['sentence'])])
        raw_data.append(sentence)
    count_vectorize.fit_transform(raw_data)
    tokenizer = count_vectorize.build_tokenizer()

    for sentence in original_sentence_list:
        splited_sentence = tokenizer(sentence['sentence'])

    word_list = count_vectorize.get_feature_names()


    for sentence in tqdm(original_sentence_list):
        written_item = {"original_sentence":sentence["sentence"],"masked_sentence":list()}
        splited_sentence = tokenizer(sentence["sentence"])
        for index in range(len(splited_sentence)-(n_gram-1)):
            phrase = ''
            count = 0
            while len(phrase.split(' '))<n_gram:
                if count+index < len(splited_sentence):
                    stemmed_word = stemmer.stem(splited_sentence[count+index])
                    phrase = phrase + ' '+ stemmed_word
                    phrase = phrase.strip()
                    count += 1
                else:
                    break
            if len(phrase.split(' ')) < n_gram:
                break
            if phrase in word_list:
                full_phrase = splited_sentence[index:index+count]
                for i in range(len(full_phrase)):
                    masked_phrase = deepcopy(full_phrase)
                    masked_phrase[i] = '[MASK]'
                    masked_sentence = splited_sentence[:index]+masked_phrase+splited_sentence[index+count:]
                    masked_sentence = ' '.join(masked_sentence)
                    written_item["masked_sentence"].append(masked_sentence)
        if written_item["masked_sentence"] != []:
            masked_sentence_list.append(written_item)
    return masked_sentence_list