#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:34:17 2021

@author: hsjomaa
"""
import time
import tensorflow as tf
import json
import openml
import numpy as np
import argparse
import os
from D2V.sampling import TestSampling, Batch
from D2V.dummdataset import Dataset_OpenML
# from D2V.modules import FunctionF, FunctionH, FunctionG, PoolF, PoolG
from D2V.extract_features_model import Dataset2VecModel
import pandas as pd

tf.random.set_seed(0)
np.random.seed(42)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--split',
                    help='Select metafeature extraction model (one can take the average of the metafeatures across all 5 splits)',
                    type=int, default=0)
parser.add_argument('--dataset', help='Select dataset by OpenML ID', type=int, default=31)
args = parser.parse_args()


rootdir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(rootdir, "D2V/checkpoints", f"split-{args.split}")
save_dir = os.path.join(rootdir, "extracted")
configuration = json.load(open(os.path.join(log_dir, "configuration.txt"), "r"))
os.makedirs(save_dir, exist_ok=True)

suite = openml.study.get_suite(99)
save_path = os.path.join(save_dir, f"{suite.name[:11]}.csv")

time0 = time.time()

for openml_dataset in suite.data:
    time1 = time.time()
    metafeatures = pd.DataFrame(data=None)
    datasetmf = []

    batch = Batch(configuration['batch_size'])
    dataset = Dataset_OpenML(openml_dataset)
    testsampler = TestSampling(dataset=dataset)

    model = Dataset2VecModel(configuration)

    model.load_weights(os.path.join(log_dir, "weights"), by_name=False, skip_mismatch=False)

    for q in range(10):  # any number of samples
        batch = testsampler.sample_from_one_dataset(batch)
        batch.collect()
        datasetmf.append(model(batch.input).numpy())

    metafeatures = np.vstack(datasetmf).mean(axis=0)[None]
    pd.DataFrame(metafeatures, index=[dataset.name]).to_csv(save_path, mode="a", header=not os.path.exists(save_path))

    time2 = time.time()

    print("Successfully finished dataset: " + dataset.name + "\nTook: " + str(np.round(time2 - time1, 2)) + " seconds")

print("Successfully finished all datasets, took: " + str(np.round(time2 - time0, 2)) + " seconds")

# Successfully finished all datasets, took: 711.93 seconds


