from transformers.training_args import TrainingArguments
from torch.utils.data import ConcatDataset, dataset
import os
from os import path
import warnings
import json
from multiprocessing import Pool
import random
from typing import Any, List, Optional, Union, Dict, Set, List
from glob import glob
from sklearn.model_selection import train_test_split

from transformers import (
    PreTrainedTokenizer,
    LineByLineWithRefDataset,
    LineByLineTextDataset,
    TextDataset,
)

from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments,  MiscArgument, ModelArguments,  get_config,  BaselineArticleMap


def get_dataset(
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
) -> Union[LineByLineWithRefDataset, LineByLineTextDataset, TextDataset, ConcatDataset]:
    basic_loss_type = training_args.loss_type.split('_')[0]
    add_loss_type = None
    if len(training_args.loss_type.split('_')) > 1:
        add_loss_type = training_args.loss_type.split('_')[1]
    if basic_loss_type == 'mlm':
        return mlm_get_dataset(data_args, tokenizer, evaluate, cache_dir)

def mlm_get_dataset(
    args: DataArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
) -> Union[LineByLineWithRefDataset, LineByLineTextDataset, TextDataset, ConcatDataset]:
    def _mlm_dataset(
        file_path: str,
        ref_path: str = None
    ) -> Union[LineByLineWithRefDataset, LineByLineTextDataset, TextDataset]:
        if args.line_by_line:
            if ref_path is not None:
                if not args.whole_word_mask or not args.mlm:
                    raise ValueError(
                        "You need to set world whole masking and mlm to True for Chinese Whole Word Mask")
                return LineByLineWithRefDataset(
                    tokenizer=tokenizer,
                    file_path=file_path,
                    block_size=args.block_size,
                    ref_path=ref_path,
                )

            return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
        else:
            return TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                overwrite_cache=args.overwrite_cache,
                cache_dir=cache_dir,
            )

    def _reformat_dataset(file_path: str) -> str:
        cache_file_path = file_path+'.cache'
        sentence_list = list()
        with open(file_path, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                item = json.loads(line.strip())
                sentence_list.append(item['original'])
                if 'augmented' in item and item['augmented'] is not None:
                    sentence_list.extend(item['augmented'])
        random.shuffle(sentence_list)
        with open(cache_file_path, mode='w', encoding='utf8') as fp:
            for sentence in sentence_list:
                fp.write(sentence+'\n')
        return cache_file_path

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    if evaluate:
        return _mlm_dataset(args.eval_data_file, args.eval_ref_file)
    elif args.train_data_files:
        return ConcatDataset([_mlm_dataset(f) for f in glob(args.train_data_files)])
    else:
        train_data_file = _reformat_dataset(args.train_data_file)
        return _mlm_dataset(train_data_file, args.train_ref_file)

def get_label_data(
    misc_args: MiscArgument,
    analysis_args: AnalysisArguments,
    data_args: DataArguments
) -> Dict[str, Dict[str, int]]:
    data_map = BaselineArticleMap()
    row_data = dict()
    for file in data_map.dataset_list:
        analysis_data_file = os.path.join(
            analysis_args.analysis_data_dir, file+'.json')
        with open(analysis_data_file) as fp:
            count = 0
            for line in fp:
                item = json.loads(line.strip())
                sentence = item['sentence']
                if sentence not in row_data:
                    row_data[sentence] = dict()
                for index, prob in item['word'].items():
                    if int(index) not in row_data[sentence]:
                        row_data[sentence][int(index)] = dict()
                    row_data[sentence][int(index)][file] = prob
                if misc_args.global_debug:
                    count += 1
                    if count == 100:
                        break

    return row_data

def main():
    misc_args, model_args, data_args, training_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, analysis_args)
    # extract_data(misc_args, data_args)


if __name__ == '__main__':
    main()
