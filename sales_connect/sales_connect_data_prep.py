from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import sales_connect.mongoSearchClient as mongoSearchClient
import sales_connect.config as C
import random
import numpy as np
import re
import os
import pandas as pd
import tensorflow as tf
import argparse
import json
from sklearn.model_selection import train_test_split

from nltk.tokenize import sent_tokenize


import warnings
warnings.filterwarnings("ignore")

def preprocess_documents(docs_df, tag_name):
    """
    Prepares the data to be used by the document generation
    :param docs: pandasdataframe
    :return:
    """
    docs = docs_df
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

def generate_samples(docs_t, labels, tag_name, n_labels, dataset_type, balance=False, max_num_sent=40):
    """
    The Balance can be:
    False = nothing is done
    True = it's balancing down to the smallest number
    int = this is an integer, there will be a sample that may balance up or down to the fixed number. The sample is done
    with replacement, so we can esily balance up with data augmentation.
    """

    # Download and extract
    tagname = tag_name
    num_labels = n_labels
    selected_labels = labels

    #     docs_t, labels = preprocess_documents(docs_df, tagname)

    #     selected_labels = list(docs_t[[tagname]].groupby(tagname).size().nlargest(num_labels).index)
    #     if len(labels) != len(selected_labels):
    #         selected_labels.append('other_remainder')

    #     tf.logging.info("This is the list of the labels we selected for tag %s: %s" % (tagname, selected_labels))
    #     docs_t[tagname] = np.where(docs_t[tagname].isin(selected_labels), docs_t[tagname], 'other_remainder')

    #     print(docs_t[[tagname]].groupby(tagname).size().nlargest(num_labels))
    minDocs = min(list(docs_t[[tagname]].groupby(tagname).size().nlargest(num_labels)))
    selected_labels = ["_".join(x.split()) for x in selected_labels]

    if balance and dataset_type == 'train':
        out = []
        if type(balance) == int:
            for label in selected_labels:
                tf.logging.info('Balancing By {}'.format(balance))
                out.append(docs_t.loc[docs_t[tagname] == label].sample(balance, replace=True))
        else:
            for label in selected_labels:
                tf.logging.info('Balancing By {}'.format(minDocs))
                out.append(docs_t.loc[docs_t[tagname] == label].sample(minDocs))
        docs_t = pd.concat(out)

    for index, row in docs_t.iterrows():
        doc = row['text']
        #         filetype = row['filetype']
        label = selected_labels.index(row[tagname])
        # label = row[tagname]

        doc = doc.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        sent_tokenize_list = sent_tokenize(doc)
        if len(sent_tokenize_list) > max_num_sent:
            sent_tokenize_list = sent_tokenize_list[:max_num_sent]
        doc = ' '.join(sent_tokenize_list)
        doc = doc.encode('ascii', errors='ignore')
        doc = doc.decode('ascii')
        doc = re.sub('\s+', ' ', doc).strip()
        yield {"inputs": doc, "label": label}

def get_dataset(data_dir, docs, tag_name='', n_labels=0, balance=None, train_percent=0.85, is_other=True):
    tagname = tag_name
    num_labels = n_labels

    docs_t, labels = preprocess_documents(docs, tagname)

    # We select the most representative labels from the dataset before splitting the dataset
    selected_labels = list(docs_t[[tagname]].groupby(tagname).size().nlargest(num_labels).index)

    tag_dir = os.path.join(data_dir, tag_name)
    if not os.path.isdir(os.path.join(data_dir, tag_name)):
        os.mkdir(tag_dir)

    labels_file = os.path.join(tag_dir, 'labels.txt')

    with tf.gfile.GFile(labels_file, "w") as f:
        f.write(str(selected_labels))

    if is_other:
        if len(labels) != len(selected_labels):
            selected_labels.append('other_remainder')
        docs_t[tagname] = np.where(docs_t[tagname].isin(selected_labels), docs_t[tagname], 'other_remainder')
    else:
        docs_t = docs_t[docs_t[tagname].isin(selected_labels)]

    tf.logging.info("This is the list of the labels we selected for tag %s: %s" % (tagname, selected_labels))
    docs_t_train_num = int(round(train_percent * len(docs_t)))

    docs_train = docs_t.iloc[:docs_t_train_num]
    docs_test = docs_t.iloc[docs_t_train_num:]

    l_train = list(generate_samples(docs_train, selected_labels, tag_name, n_labels, 'train', balance=False))
    list_data_train_ = random.sample(l_train, len(l_train))

    l_test = list(generate_samples(docs_test, selected_labels, tag_name, n_labels, 'test', balance=False))
    list_data_test_ = random.sample(l_test, len(l_test))

    tag_dir = os.path.join(data_dir, tag_name)
    if not os.path.isdir(os.path.join(data_dir, tag_name)):
        os.mkdir(tag_dir)

    dataset_train_file_name = 'train.json'
    dataset_train_file = os.path.join(tag_dir, dataset_train_file_name)

    dataset_test_file_name = 'test.json'
    dataset_test_file = os.path.join(tag_dir, dataset_test_file_name)

    dataset_dev_file_name = 'dev.json'
    dataset_dev_file = os.path.join(tag_dir, dataset_dev_file_name)


    with tf.gfile.GFile(dataset_train_file, "w") as f:
        json.dump(list_data_train_, f)

    with tf.gfile.GFile(dataset_test_file, "w") as f:
        json.dump(list_data_test_, f)

    with tf.gfile.GFile(dataset_dev_file, "w") as f:
        json.dump(list_data_test_, f)

    return 1

def parse_args():
    description = ('')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_dir', help='This is the score file we want to use to compute the perplexity.')
    args = parser.parse_args()
    return args

def main():
    pwd = os.getcwd()
    # path_ = os.path.join(pwd, '.gcloud_sa_cred/csmlexp-dev-gkjc-07defcba37aa.json')
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path_
    #
    # storage_client = storage.Client()
    # bucket_name = 'ml_models_storage'
    # bucket = storage_client.get_bucket(bucket_name)

    MONGOSC = mongoSearchClient.MongoSalesConnect(C.SALESCONNECT_MONGO_URL, C.SALESCONNECT_DB).connect()
    MONGOSC.setCollection(C.POLLED_COLL)


    collection = MONGOSC.coll
    all_ = collection.find({"text": {"$ne": ''}})
    docs = pd.DataFrame(list(all_))

    tagname = 'bizent'
    num_labels = 5
    test_percent = 0.2

    args = parse_args()
    data_dir = args.data_dir

    tags = [ 'bizent', 'contentcategory', 'contentsubcategory', 'subbizent', 'prodfam' ]

    for tagname in tags:
        tf.logging.info("Doing the tagging for tag: %s" % tagname)
        get_dataset(data_dir, docs, tag_name=tagname, n_labels=5, balance=None, train_percent=0.85, is_other=False)

if __name__ == '__main__':
    main()

