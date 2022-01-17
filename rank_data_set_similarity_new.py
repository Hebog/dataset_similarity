import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.metrics.pairwise import cosine_similarity
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
from MFE.extract_features import extract_features_OpenML

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dataset', help='Select did of dataset to base ranking on', type=int, default=14)
parser.add_argument('--split',
                    help='Select metafeature extraction model (one can take the average of the metafeatures across all 5 splits)',
                    type=int, default=0)
args = parser.parse_args()

openml_dataset = args.input_dataset

# Extract Metafeatures for input dataset

# Process D2V features

rootdir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(rootdir, "D2V/checkpoints", f"split-{args.split}")
configuration = json.load(open(os.path.join(log_dir, "configuration.txt"), "r"))

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
mf_df = pd.DataFrame(metafeatures, index=[openml_dataset])
mf_df.insert(0, "dataset_name", value=dataset.name)
mf_df = mf_df.loc[openml_dataset]

# Process MFE features
name, metafeatures_mfe = extract_features_OpenML(openml_dataset)

mfe_extracted = True if isinstance(metafeatures_mfe, pd.DataFrame) else False
if mfe_extracted:
    metafeatures_mfe.insert(0, "dataset_name", value=name)
    metafeatures_mfe = metafeatures_mfe.loc[openml_dataset]


def create_ranking(openml_dataset, extr_mf_path, input_mf):
    extracted_mf = pd.read_csv(extr_mf_path, index_col=0)
    if openml_dataset in extracted_mf.index:
        print("Input index already in features")

    else:
        extracted_mf = extracted_mf.append(input_mf)

    min_max_scaler = MinMaxScaler()
    mf_scaled = min_max_scaler.fit_transform(extracted_mf.iloc[:, 1:])
    mf_scaled = pd.DataFrame(mf_scaled, index=extracted_mf.index)
    # print("Initial Shape: " + str(mf_scaled.shape))
    na_count_cols = mf_scaled.isna().sum() / len(mf_scaled)
    mf_scaled = mf_scaled.loc[:, na_count_cols <= 0.20]
    inp_ar = np.array(mf_scaled.loc[openml_dataset, :]).reshape(1, -1)
    nan_col_inp = np.argwhere(np.isnan(inp_ar))[:, 1]
    inp_ar = np.delete(inp_ar, nan_col_inp, axis=1)
    filtered = mf_scaled[mf_scaled.index != openml_dataset]
    comp_ind = filtered.index
    filtered = np.delete(np.asarray(filtered), nan_col_inp, axis=1)
    comp_ar = filtered[~np.isnan(filtered).any(axis=1), :]
    comp_ind = comp_ind[~np.isnan(filtered).any(axis=1)]

    # print("Final Shape: " + str(comp_ar.shape))

    cos_sim_ar = np.zeros((len(comp_ar), 1))
    for i in range(len(comp_ar)):
        cos_sim_ar[i] = cosine_similarity(comp_ar[i].reshape(1,-1), inp_ar)

    cos_sim_df = pd.DataFrame({"did":comp_ind, "Cosine Similarity": cos_sim_ar.reshape(-1)})
    cos_sim_df['dataset_name'] = extracted_mf[extracted_mf.index != openml_dataset].reset_index()['dataset_name']
    ranking = cos_sim_df.sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)
    return ranking


ranking = create_ranking(openml_dataset, "extracted_MF/OpenML-CC18_mfe.csv", metafeatures_mfe)
ranking2 = create_ranking(openml_dataset, "extracted_MF/OpenML-CC18_d2v.csv", mf_df)

print("Similarity ranking based on MFE metafeatures:")
print(ranking.head(10))
print("\n\n")
print("Similarity ranking based on D2V metafeatures:")
print(ranking2.head(10))

