# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IMDB Sentiment Classification Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import csv
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import pandas as pd
import zipfile

from nltk.tokenize import word_tokenize

import argparse

import tensorflow as tf

import json
import re

def parse_args():
    description = ('')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_dir', help='This is the score file we want to use to compute the perplexity.')
    parser.add_argument('-t', '--tmp_dir', help='This is a temporary directory.')
    args = parser.parse_args()
    return args


@registry.register_problem
class Sentiment140(text_problems.Text2ClassProblem):
    """IMDB sentiment classification."""
    URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    train_file_name = 'training.1600000.processed.noemoticon.csv'
    test_file_name = 'testdata.manual.2009.06.14.csv'

    @property
    def is_generate_per_split(self):
        return True

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 10,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def approx_vocab_size(self):
        return 2 ** 13  # 8k vocab suffices for this small dataset.

    @property
    def num_classes(self):
        return 2

    def class_labels(self, data_dir):
        del data_dir
        return ["neg", "pos"]

    def doc_generator(self, sentiment140_dir, dataset, include_label=False):
        dirs = [(os.path.join(sentiment140_dir, dataset, "pos"), True), (os.path.join(sentiment140_dir, dataset, "neg"), False)]

        for d, label in dirs:
            for filename in os.listdir(d):
                with tf.gfile.Open(os.path.join(d, filename)) as imdb_f:
                    doc = imdb_f.read().strip()
                    if include_label:
                        yield doc, label
                    else:
                        yield doc

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate examples."""
        # Download and extract
        compressed_filename = os.path.basename(self.URL)
        download_path = generator_utils.maybe_download(tmp_dir, compressed_filename, self.URL)
        # sentiment140_dir = os.path.join(tmp_dir, "sentiment140")
        if not tf.gfile.Exists(tmp_dir):
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

        # Generate examples
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "train" if train else "test"
        # punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

        punctuation = '!#$%&\()*+-/:;<=>?@[\\]^_`{|}'

        if dataset_split == 'train':
            file_name = os.path.join(tmp_dir, self.train_file_name)
        else:
            file_name = os.path.join(tmp_dir, self.test_file_name)

        df = pd.read_csv(file_name, header=None, usecols=[0, 5], encoding='latin-1')
        # Remove URLs and @ stuff
        df[5] = df[5].apply(lambda x: re.sub(r"http\S+", "", x))
        df[5] = df[5].apply(lambda x: re.sub(r"@\S+", "",  x))
        df[5] = df[5].apply(lambda x: re.sub(r"#\S+", "",  x))

        for i, r in df.iterrows():
            text = r[5]
            label = r[0]
            tokens = word_tokenize(text)
            table = str.maketrans('', '', punctuation)
            stripped = [w.translate(table) for w in tokens]
            tokens = [word.lower() for word in stripped]
            # 512 is the max len for thje
            # if len(tokens) > 512 :
            #     tokens = tokens[:512]
            # tokens = [word for word in stripped]
            doc = ' '.join(tokens)
            doc = re.sub('\'\'', '', doc)
            doc = re.sub('\s+', ' ', doc)
            if int(label) in ([0, 4]):
                yield {
                    "inputs": doc,
                    "label": int(label),
                }

def main():
    args = parse_args()  # units_file_name = args.units_dir  #transcriptions_file_name = args.scores_dir
    DATA_DIR = args.data_dir
    TMP_DIR = args.tmp_dir

    sentiment140_problem = Sentiment140()

    # imdb_problem.generate_data(DATA_DIR, TMP_DIR)

    gen_data_train = sentiment140_problem.generate_samples(DATA_DIR, TMP_DIR, 'train')
    gen_data_test = sentiment140_problem.generate_samples(DATA_DIR, TMP_DIR, 'test')

    dataset_train_file = os.path.join(DATA_DIR, 'train.json')
    dataset_test_file = os.path.join(DATA_DIR, 'test.json')

    with tf.gfile.GFile(dataset_train_file, "w") as f:
        json.dump(list(gen_data_train), f)

    with tf.gfile.GFile(dataset_test_file, "w") as f:
        json.dump(list(gen_data_test), f)

if __name__ == '__main__':
    main()
