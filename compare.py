import os
import json
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
import numpy as np
import tqdm
from dataclasses import dataclass, field
from typing import List, Dict
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class BaselineArticleMap:
    dataset_to_name: Dict = field(default_factory=lambda: {'Breitbart':'Breitbart','CBS':'CBS News','CNN':'CNN','Fox':'Fox News','HuffPost':'HuffPost','NPR':'NPR','NYtimes':'New York Times','usatoday':'USA Today','wallstreet':'Wall Street Journal','washington':'Washington Post'})
    name_to_dataset: Dict = field(init=False)
    dataset_list: List[str] = field(init=False)
    dataset_bias: Dict = field(default_factory=lambda:{'Breitbart':2,'CBS':-1,'CNN':-5/3,'Fox':5/3,'HuffPost':-2,'NPR':-0.5,'NYtimes':-1.5,'usatoday':-1,'wallstreet':0.5,'washington':-1})
    left_dataset_list: List[str] = field(default_factory=lambda:['Breitbart', 'Fox', 'sean','rushlimbaugh.com'])

    def __post_init__(self):
        self.name_to_dataset = {v: k for k, v in self.dataset_to_name.items()}
        self.dataset_list = [k for k,v in self.dataset_to_name.items()]

def lda_baseline(mean_method, file_list):
    def bd(vector_a, vector_b):
        bc = np.sum(
            np.sqrt(vector_a * vector_b))
        distance = -np.log(bc)
        return distance
    data_path = "./data/"
    topic_list = ["climate-change", "corporate-tax", "drug-policy", "gay-marriage", "obamacare"]
    distance_dict = dict()
    for topic in topic_list:
        if mean_method == 'average':
            common_text = list()
            outlets_text_list = [[] for _ in range(10)]
            topic_path = os.path.join(data_path,'data_'+topic)
            topic_path = topic_path + '/42/all/original'
            for file_path in file_list:
                file = topic_path+'/'+file_path
                with open(file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        text = item["sentence"].split(' ')
                        label = item["label"]
                        outlets_text_list[int(label)].append(text)
                        common_text.append(text)
                    common_dictionary = Dictionary(common_text)

            common_corpus = [common_dictionary.doc2bow(text) for text in common_text]
            outlets_text_corpus = list()
            for outlets_text in outlets_text_list:
                outlets_text_corpus.append([common_dictionary.doc2bow(text) for text in outlets_text])
            print("LDA running")
            n_topic = 10
            lda = LdaMulticore(common_corpus, num_topics=n_topic,random_state=42,workers=4,passes=2)
            print("LDA finish")
            outlets_vec_list = list()
            for outlets in tqdm.tqdm(outlets_text_corpus):
                t_list = list()
                for t in outlets:
                    outlets_vec_temp = lda[t]
                    outlets_vec = list(0 for _ in range(n_topic))
                    for item in outlets_vec_temp:
                        outlets_vec[item[0]] = item[1]
                    for i,_ in enumerate(outlets_vec):
                        if outlets_vec[i] == 0:
                            outlets_vec[i] = 1e-10
                    t_list.append(np.array(outlets_vec))
                t_distance  = np.mean(np.array(t_list),axis=0)
                outlets_vec_list.append(t_distance)
        elif mean_method == 'combine':
            common_text = list()
            outlets_text_list = [[] for _ in range(10)]
            topic_path = os.path.join(data_path,'data_'+topic)
            topic_path = topic_path + '/42/all/original'
            for file_path in file_list:
                file = topic_path+'/'+file_path
                with open(file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        text = item["sentence"].split(' ')
                        label = item["label"]
                        outlets_text_list[int(label)].extend(text)
            for text in outlets_text_list:
                common_text.append(text)
            common_dictionary = Dictionary(common_text)

            common_corpus = [common_dictionary.doc2bow(text) for text in common_text]

            print("LDA running")
            n_topic = 10
            lda = LdaMulticore(common_corpus, num_topics=n_topic,random_state=42,workers=4,passes=2)
            print("LDA finish")
            outlets_vec_list = list()
            for outlets in common_corpus:
                outlets_vec_temp = lda[outlets]
                outlets_vec = list(0 for _ in range(n_topic))
                for item in outlets_vec_temp:
                    outlets_vec[item[0]] = item[1]
                for i,_ in enumerate(outlets_vec):
                    if outlets_vec[i] == 0:
                        outlets_vec[i] = 1e-10
                outlets_vec_list.append(np.array(outlets_vec))       

        distance_matrix = []
        for i, outlets_a_vec in enumerate(outlets_vec_list):
            d_list = [0 for _ in range(len(outlets_vec_list))]
            for j, outlets_b_vec  in enumerate(outlets_vec_list):
                if i!=j:
                    # distance = entropy(topic_b_vec,topic_a_vec)
                    # distance = bd(outlets_b_vec,outlets_a_vec)
                    distance = cosine_distances(outlets_b_vec.reshape(1, -1),outlets_a_vec.reshape(1, -1))[0][0]
                    d_list[j] = distance
            distance_matrix.append(np.array(d_list))
        distance_matrix = np.array(distance_matrix)
        distance_dict[topic] = distance_matrix
    return distance_dict

def tfidf_baseline(mean_method, file_list):
    data_path = "./data/"
    topic_list = ["climate-change", "corporate-tax", "drug-policy", "gay-marriage", "obamacare"]
    distance_dict = dict()
    for topic in topic_list:
        if mean_method == 'average':
            common_text = list()
            outlets_text_list = [[] for _ in range(10)]
            topic_path = os.path.join(data_path,'data_'+topic)
            topic_path = topic_path + '/42/all/original'
            for file_path in file_list:
                file = topic_path+'/'+file_path
                with open(file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        text = item["sentence"].split(' ')
                        label = item["label"]
                        outlets_text_list[int(label)].append(' '.join(text))
                        common_text.append(' '.join(text))
            vectorizer = TfidfVectorizer(ngram_range=(1,3))
            model = vectorizer.fit(common_text)


            outlets_vec_list = list()
            for outlets in outlets_text_list:
                vec = model.transform(outlets)
                vec = np.mean(vec,axis=0)
                outlets_vec_list.append(vec)

        elif mean_method == 'combine':
            common_text = list()
            outlets_text_list = [[] for _ in range(10)]
            topic_path = os.path.join(data_path,'data_'+topic)
            topic_path = topic_path + '/42/all/original'
            for file_path in file_list:
                file = topic_path+'/'+file_path
                with open(file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        text = item["sentence"].split(' ')
                        label = item["label"]
                        outlets_text_list[int(label)].extend(text)
            for text in outlets_text_list:
                common_text.append(' '.join(text))

            vectorizer = TfidfVectorizer()
            outlets_vec_list = vectorizer.fit_transform(common_text)
            outlets_vec_list = outlets_vec_list.todense()

        distance_matrix = []
        for i, outlets_a_vec in enumerate(outlets_vec_list):
            d_list = [0 for _ in range(len(outlets_vec_list))]
            for j, outlets_b_vec  in enumerate(outlets_vec_list):
                if i!=j:
                    # disance = entropy(topic_b_vec,topic_a_vec)
                    disance = cosine_distances(outlets_b_vec,outlets_a_vec)
                    d_list[j] = disance[0][0]
            distance_matrix.append(np.array(d_list))
        distance_matrix = np.array(distance_matrix)
        distance_dict[topic] = distance_matrix
    return distance_dict

def get_baseline(ground_truth_list, file_list, method, combine_method):
    data_map = BaselineArticleMap()
    bias_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)))
    if method == "tfidf":
        baseline_matrix_list = tfidf_baseline(combine_method,file_list)
    elif method == "lda":
        baseline_matrix_list = lda_baseline(combine_method,file_list)

    for ground_truth in ground_truth_list:
        if ground_truth == "MBR":
            ground_truth_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.float32)
            ground_truth_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)
            for i,media_a in enumerate(data_map.dataset_list):
                temp_distance = list()
                for j,media_b in enumerate(data_map.dataset_list):
                    bias_distance_matrix[i][j] = abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b])
                    temp_distance.append(abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b]))
                    ground_truth_distance_matrix[i][j] = abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b])
                distance_set = set(temp_distance)
                distance_set = sorted(list(distance_set))
                for o, d_o in enumerate(distance_set):
                    for j,d_j in enumerate(temp_distance):
                        if d_o == d_j:
                            ground_truth_distance_order_matrix[i][j] = o
        elif ground_truth in ['SoA-s','SoA-t']:
            ground_truth_distance_matrix = np.load('./log/ground-truth/model/ground-truth_'+ground_truth+'.npy')
            ground_truth_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)
            for i,media_a in enumerate(data_map.dataset_list):
                temp_distance = ground_truth_distance_matrix[i]
                distance_set = set(temp_distance)
                distance_set = sorted(list(distance_set))
                for o, d_o in enumerate(distance_set):
                    for j,d_j in enumerate(temp_distance):
                        if d_o == d_j:
                            ground_truth_distance_order_matrix[i][j] = o
        
        groundtruth_file = './analysis/ground-truth/round-truth_'+method+'_'+combine_method+'.json'
        performace_dict =  {'topic':'average','ground_truth':ground_truth,'tau_performance':[]}
        for topic, media_distance in baseline_matrix_list.items():

            media_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)
            for i,media_a in enumerate(data_map.dataset_list):
                temp_distance = list()
                for j,media_b in enumerate(data_map.dataset_list):
                    temp_distance.append(media_distance[i][j])
                order_list = np.argsort(temp_distance)
                order_list = order_list.tolist()
                for j in range(len(data_map.dataset_list)):
                    order = order_list.index(j)
                    media_distance_order_matrix[i][j] = order
            media_count = len(data_map.dataset_list)
            
            
            
            
            tau_performance = 0
            for i in range(media_count):
                tau, p_value = kendalltau(media_distance_order_matrix[i].reshape(1,-1), ground_truth_distance_order_matrix[i].reshape(1,-1))
                tau_performance += tau
            tau_performance /= media_count


            record_item = {'topic':topic,'ground_truth':ground_truth,'tau_performance':round(tau_performance,2)}

            performace_dict['tau_performance'].append(round(tau_performance,2))
            with open(groundtruth_file,mode='a',encoding='utf8') as fp:
                fp.write(json.dumps(record_item,ensure_ascii=False)+'\n')
        
        performace_dict['tau_performance'] = str(round(np.mean(performace_dict['tau_performance']),2)) + "("+str(round(np.std(performace_dict['tau_performance'],ddof=1),2))+")"
        with open(groundtruth_file,mode='a',encoding='utf8') as fp:
            fp.write(json.dumps(performace_dict,ensure_ascii=False)+'\n')


def main():
    for file_list in [['en.valid']]:
        for method in ['tfidf','lda']:
            for combine_method in ["average"]:
                get_baseline(['SoA-t','SoA-s','MBR'], file_list, method, combine_method)

if __name__ == '__main__':
    main()