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
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

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
class SentimentIMDB(text_problems.Text2ClassProblem):
    """IMDB sentiment classification."""
    URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    MAX_LEN_SENT = 10000

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

    def doc_generator(self, imdb_dir, dataset, include_label=False):
        dirs = [(os.path.join(imdb_dir, dataset, "pos"), True), (os.path.join(
            imdb_dir, dataset, "neg"), False)]

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
        download_path = generator_utils.maybe_download(tmp_dir, compressed_filename,
                                                       self.URL)
        imdb_dir = os.path.join(tmp_dir, "aclImdb")
        if not tf.gfile.Exists(imdb_dir):
            with tarfile.open(download_path, "r:gz") as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, tmp_dir)

        # Generate examples
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "train" if train else "test"
        # punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        punctuation = '!#$%&\()*+-/:;.<=>?@[\\]^_`{|}'
        for doc, label in self.doc_generator(imdb_dir, dataset, include_label=True):
            sentences = sent_tokenize(doc)
            # if len(sentences) > 3:
            #     sentence = sentences[:3]
            #
            # else:
            #     sentence = sentences
            #
            # sentence = " ".join(sentence)
            sentence = sentences[:]
            sentence = ' '.join(sentence)

            tokens = word_tokenize(sentence)
            table = str.maketrans('', '', punctuation)
            stripped = [w.translate(table) for w in tokens]
            tokens = [word.lower() for word in stripped if word != 'br']
            # 512 is the max len for the bert models
            if len(tokens) < self.MAX_LEN_SENT:
                # tokens = [word for word in stripped]
                doc = ' '.join(tokens)
                doc = re.sub('\'\'', '', doc).strip()
                doc = re.sub('\s+', ' ', doc).strip()
                yield {
                    "inputs": doc,
                    "label": int(label),
                }



def main():
    args = parse_args()  # units_file_name = args.units_dir  #transcriptions_file_name = args.scores_dir
    DATA_DIR = args.data_dir
    TMP_DIR = args.tmp_dir

    imdb_problem = SentimentIMDB()

    # imdb_problem.generate_data(DATA_DIR, TMP_DIR)

    gen_data_train = imdb_problem.generate_samples(DATA_DIR, TMP_DIR, 'train')
    gen_data_test = imdb_problem.generate_samples(DATA_DIR, TMP_DIR, 'test')

    dataset_train_file = os.path.join(DATA_DIR, 'train.json')
    dataset_test_file = os.path.join(DATA_DIR, 'test.json')

    with tf.gfile.GFile(dataset_train_file, "w") as f:
        json.dump(list(gen_data_train), f)

    with tf.gfile.GFile(dataset_test_file, "w") as f:
        json.dump(list(gen_data_test), f)

if __name__ == '__main__':
    main()
