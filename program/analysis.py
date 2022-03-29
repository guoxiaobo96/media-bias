import os
from random import random
import csv
from dataclasses import dataclass, field
from nltk.stem.porter import *
from abc import ABC, abstractmethod
from glob import glob
from tqdm import tqdm
from typing import Any, Dict, Optional, Set, Tuple, Union, List
from grakel.graph_kernels import *
from scipy.stats import kendalltau
import random
from sklearn.metrics.pairwise import(
    cosine_distances,
)
import numpy as np
from numpy import ndarray


from .config import AnalysisArguments, MiscArgument, ModelArguments, DataArguments, TrainingArguments


class BaseAnalysis(ABC):
    def __init__(
        self,
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        config: AnalysisArguments

    ) -> None:
        self._config = config
        self._encoder = None
        self._analyser = None
        self._misc_args = misc_args
        self._data_args = data_args
        self._model_args = model_args
        self._training_args = training_args

        self._load_encoder(self._config.analysis_encode_method, misc_args,
                           model_args, data_args, training_args)

    def _load_encoder(
        self,
        encode_method: str,
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments
    ) -> None:
        if encode_method == "term":
            self._encoder = TermEncoder()

    @abstractmethod
    def _load_analysis_model(
        self,
        compare_method: str
    ):
        pass

    @abstractmethod
    def analyze(
        self,
        data,
        sentence_number: str,
        analysis_args: AnalysisArguments
    ):
        pass

    def _encode_data(
        self,
        data
    ) -> Tuple[List[str], List[ndarray]]:
        encoded_result = self._encoder.encode(data)
        dataset_list = list(encoded_result.keys())
        encoded_list = list(encoded_result.values())
        return dataset_list, encoded_list

class CorrelationAnalysis(BaseAnalysis):
    def __init__(self,  misc_args: MiscArgument, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, config: AnalysisArguments) -> None:
        super().__init__(misc_args, model_args, data_args, training_args, config)
        self._load_analysis_model(self._config.analysis_correlation_method)

    def _load_analysis_model(
        self,
        method: str
    ):
        if method == "tau":
            self._analyser = self._rank_distance
        elif method == 'pearson':
            self._analyser = cosine_distances



    def _rank_distance(self,data,data_map):
        distance_matrix = cosine_distances(data)
        media_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int)
        for i,media_a in enumerate(data_map.dataset_list):
            temp_distance = list()
            distance_map = list()
            for j,media_b in enumerate(data_map.dataset_list):
                distance_map.append((j,distance_matrix[i][j]))
            random.seed(42)
            random.shuffle(distance_map)
            for item in distance_map:
                temp_distance.append(item[1])
            # if temp_distance.count(0) > 1:
            #     return None
            order_list = np.argsort(temp_distance)
            order_list = order_list.tolist()
            
            for order, v in enumerate(order_list):
                media_distance_order_matrix[i][distance_map[v][0]] = order

            # order_list = np.argsort(temp_distance)
            # order_list = order_list.tolist()
            # for j in range(len(data_map.dataset_list)):
            #     order = distance_map[order_list.index(j)][1]
            #     media_distance_order_matrix[i][j] = order
        return media_distance_order_matrix


    

    def analyze(
        self,
        data,
        sentence_number: str,
        analysis_args: AnalysisArguments,
        keep_result=True,
        encode: bool = True,
        dataset_list: List = [],
        data_map = None,

    ) -> Dict[int, Set[str]]:
        cluster_result = dict()
        if encode:
            if 'vanilla' in data:
                data.pop('vanilla')
            dataset_list, encoded_list = self._encode_data(data)
        else:
            encoded_list = data
        if self._config.analysis_correlation_method == 'tau':
            media_distance_matrix = self._analyser(encoded_list,data_map)
        elif self._config.analysis_correlation_method == 'pearson':
            media_distance_matrix = self._analyser(encoded_list)
        return media_distance_matrix, media_distance_matrix, dataset_list, encoded_list

class TermEncoder(object):
    def __init__(self) -> None:
        self._term_dict = dict()

    def encode(
        self,
        data: Dict
    ) -> Dict[str, Dict]:
        term_set = set()
        encode_result = dict()
        for _, term_dict in data.items():
            term_set = term_set.union(set(term_dict.keys()))
        for i, term in enumerate(list(term_set)):
            self._term_dict[term] = i
        for dataset, term_dict in data.items():
            encode_array = np.zeros(shape=len(term_set))
            for k, v in term_dict.items():
                encode_array[self._term_dict[k]] = float(v)
            encode_result[dataset] = encode_array
        return encode_result

class CorrelationCompare(object):
    def __init__(self, misc_args: MiscArgument, analysis_args: AnalysisArguments) -> None:
        super().__init__()
        self. _result_path = os.path.join(os.path.join(
            analysis_args.analysis_result_dir, analysis_args.graph_distance), analysis_args.graph_kernel)
        self._analysis_args = analysis_args
    def compare(self, model_dict):
        name_list = list()
        model_list = list()
        result_dict = dict()

        for name, model in model_dict.items():
            name_list.append(name)
            model_list.append(model)

        base_index = name_list.index('base')
        step_size = 0.05
        distribution = [0 for _ in range(int(2/step_size) + 1)]
        for k, name in enumerate(tqdm(name_list,desc="Calculate distance")):
            if k == base_index:
                continue
            performance = 0
            if  self._analysis_args.analysis_correlation_method == 'tau':
                for i in range(len(model_list[base_index])):
                    tau, p_value = kendalltau(model_list[k][i].reshape(1,-1), model_list[base_index][i].reshape(1,-1))
                    performance += tau
            elif self._analysis_args.analysis_correlation_method == 'pearson':
                for i in range(len(model_list[base_index])):
                    pearson = np.corrcoef(model_list[k][i].reshape(1,-1),model_list[base_index][i].reshape(1,-1))
                    performance += pearson[0][1]
            performance /= len(model_list[base_index])
            result_dict[name] = performance
        return result_dict

def main():
    # from config import get_config
    # from data import get_analysis_data
    # from util import prepare_dirs_and_logger
    # misc_args, model_args, data_args, training_args, analysis_args = get_config()
    # prepare_dirs_and_logger(misc_args, model_args,
    #                         data_args, training_args, analysis_args)
    # analysis_data = get_analysis_data(analysis_args)
    # analysis_model = DistanceAnalysis(model_args, data_args, training_args, analysis_args)
    # for k,v in analysis_data.items():
    #     analysis_model.analyze(analysis_data['4.json'])
    log_dir = '../../log/tweets'
    category_file = os.path.join(os.path.join(log_dir, 'dict'), 'category.csv')
    with open(category_file, mode='r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            if 'Word' in row:
                continue
            else:
                category = [0 for _ in range(len(row)-1)]
                for i, mark in enumerate(row):
                    if mark == 'X':
                        category[i-1] = 1


if __name__ == '__main__':
    main()
