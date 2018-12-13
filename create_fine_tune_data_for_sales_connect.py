from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from nltk.tokenize import sent_tokenize

import os
import re
import argparse
from tensor2tensor.data_generators import generator_utils

import tensorflow as tf

import json

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

URL = 'https://cisco.box.com/shared/static/ju3bkz4w042gyfpf2kqpc38f2s19isu1.pkl'

URL_TRAIN = 'https://cisco.box.com/shared/static/i35lgkfw11i1ktg0y899jkothjlfube7.pkl'
URL_TEST = 'https://cisco.box.com/shared/static/5nqpyltnap70wqpr707wp944ubs1rkkt.pkl'
URL_DEV = 'https://cisco.box.com/shared/static/5nqpyltnap70wqpr707wp944ubs1rkkt.pkl'


def parse_args():
    description = ('')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_dir', help='This is the score file we want to use to compute the perplexity.')
    parser.add_argument('-t', '--tmp_dir', help='This is a temporary directory.')
    args = parser.parse_args()
    return args

def preprocess_documents(tmp_dir, tag_name, url):
    """
    Prepares the data to be used by the document generation
    :param docs: pandasdataframe
    :return:
    """
    URL = url
    compressed_filename = os.path.basename(URL)
    download_path = generator_utils.maybe_download(tmp_dir, compressed_filename, URL)
    docs = pd.read_pickle(download_path)

    docs_c = docs[[tag_name, 'text']]
    docs_c.dropna(inplace=True)
    s = docs_c[tag_name].apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = tag_name
    del docs_c[tag_name]
    docs_c = docs_c.join(s)
    docs_c[tag_name] = docs_c[tag_name].apply(lambda x: "_".join(x.split()))
    docs_c.reset_index(drop=True, inplace=True)
    docs_c['text'] = docs_c['text'].apply(lambda x: x[0])
    # docs_c[tag_name] = docs_c[tag_name].apply(lambda x: "_".join(x.split()))
    labels = sorted(docs_c[tag_name].unique())
    return docs_c, labels

def generate_samples(data_dir, tmp_dir, tag_name, num_labels, dataset_type):
    """Generate samples of text and label pairs.

    Each yielded dict will be a single example. The inputs should be raw text.
    The label should be an int in [0, self.num_classes).

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).

    Yields:
      {"inputs": text, "label": int}
    """
    MAX_NUM_SENT = 40

    # if dataset_type == 'train':
    #     URL = URL
    # elif dataset_type == 'test':
    #     URL = URL
    # else:
    #     URL = URL

    docs_t, labels = preprocess_documents(tmp_dir, tag_name, URL)
    selected_labels = list(docs_t[[tag_name]].groupby(tag_name).size().nlargest(num_labels).index)

    selected_labels = [ "_".join(x.split()) for x in selected_labels ]

    tf.logging.info("This is the list of the labels we selected for tag %s: %s" % (tag_name, selected_labels))

    docs_t = docs_t[docs_t[tag_name].isin(selected_labels)]

    tag_dir = os.path.join(data_dir, tag_name)
    if not os.path.isdir(os.path.join(data_dir, tag_name)):
        os.mkdir(tag_dir)

    labels_file = os.path.join(tag_dir, 'labels.txt')

    with tf.gfile.GFile(labels_file, "w") as f:
        f.write(str(selected_labels))

    examples = []

    for index, row in docs_t.iterrows():
        doc = row['text']
        label = selected_labels.index(row[tag_name])
        doc = doc.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        sent_tokenize_list = sent_tokenize(doc)
        if len(sent_tokenize_list) > MAX_NUM_SENT:
            sent_tokenize_list = sent_tokenize_list[:MAX_NUM_SENT]
        doc = ' '.join(sent_tokenize_list)
        doc = doc.encode('ascii', errors='ignore')
        doc = doc.decode('ascii')
        doc = re.sub('\s+', ' ', doc).strip()
        examples.append({"text": doc, "label": label})

    dataset_file_name = dataset_type+'.json'

    dataset_file = os.path.join(tag_dir, dataset_file_name)

    with tf.gfile.GFile(dataset_file, "w") as f:
        json.dump(examples, f)
    return 1

def main():
    args = parse_args()  #units_file_name = args.units_dir  #transcriptions_file_name = args.scores_dir
    DATA_DIR = args.data_dir
    TMP_DIR = args.tmp_dir
    tag_name = 'contentcategory'
    docs_c, labels = preprocess_documents(TMP_DIR, tag_name, URL)

    tags = ['bizent', 'bizentity', 'contentcategory', 'contentsubcategory', 'subbizent']

    for tag_name in tags:
        tf.logging.info("Doing the tagging for tag: %s" % tag_name)
        data_generator = generate_samples(DATA_DIR, TMP_DIR, tag_name, len(labels), 'train')

if __name__ == '__main__':
    main()