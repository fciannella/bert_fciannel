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

import argparse

import tensorflow as tf

import json


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
                tar.extractall(tmp_dir)

        # Generate examples
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "train" if train else "test"
        for doc, label in self.doc_generator(imdb_dir, dataset, include_label=True):
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
