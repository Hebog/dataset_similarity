import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# from D2V.dummdataset import Dataset_OpenML_D2V
# from D2V.extract_features_model import Dataset2VecModel
# from D2V.sampling import TestSampling, Batch
# from MFE.extract_features import extract_MFE_features_OpenML


# Extract Metafeatures for input dataset

# Process D2V features
def process_d2v(openml_did, split=0):
    rootdir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(rootdir, "D2V/checkpoints", f"split-{split}")
    configuration = json.load(open(os.path.join(log_dir, "configuration.txt"), "r"))

    metafeatures = pd.DataFrame(data=None)
    datasetmf = []

    batch = Batch(configuration['batch_size'])
    dataset = Dataset_OpenML_D2V(openml_did)
    testsampler = TestSampling(dataset=dataset)

    model = Dataset2VecModel(configuration)

    model.load_weights(os.path.join(log_dir, "weights"), by_name=False, skip_mismatch=False)

    for q in range(10):  # any number of samples
        batch = testsampler.sample_from_one_dataset(batch)
        batch.collect()
        datasetmf.append(model(batch.input).numpy())

    metafeatures = np.vstack(datasetmf).mean(axis=0)[None]
    metafeatures_d2v = pd.DataFrame(metafeatures, index=[openml_did])
    metafeatures_d2v.columns = metafeatures_d2v.columns.astype(str)  # Ensure the columns have string names
    metafeatures_d2v.insert(0, "dataset_name", value=dataset.name)
    return metafeatures_d2v


# Process MFE features
def process_mfe(openml_did):
    # As the MFE features tend to have errors with some datasets, the extracting is in a try/except
    try:
        name, metafeatures_mfe = extract_MFE_features_OpenML(openml_did)
        mfe_extracted = True if isinstance(metafeatures_mfe, pd.DataFrame) else False

    except:
        mfe_extracted = False
        print("MFE features could not be extracted")

    if mfe_extracted:
        metafeatures_mfe.insert(0, "dataset_name", value=name)
        # metafeatures_mfe = metafeatures_mfe.loc[openml_dataset]

        return metafeatures_mfe, mfe_extracted

    else:
        return None, mfe_extracted


def create_ranking(openml_dataset, extr_mf_path, input_mf=None):
    extracted_mf = pd.read_csv(extr_mf_path, index_col=0)
    if openml_dataset in extracted_mf.index:
        print("Input index already in features")

    else:
        assert input_mf is not None, "No input index in features and not MF input"
            
        extracted_mf = extracted_mf.append(input_mf)

    # Checking whether any value if infity, if so, removing it
    to_be_scaled_df = extracted_mf.iloc[:, 1:]
    to_be_scaled_df = to_be_scaled_df.loc[:, ~np.isinf(to_be_scaled_df).any()]

    # Scaling the metafeature dataframe
    min_max_scaler = MinMaxScaler()
    mf_scaled = min_max_scaler.fit_transform(to_be_scaled_df)
    mf_scaled = pd.DataFrame(mf_scaled, index=extracted_mf.index)

    # print("Initial Shape: " + str(mf_scaled.shape))
    # Checking if some columns have high NA count (only applicable for MFE)
    na_count_cols = mf_scaled.isna().sum() / len(mf_scaled)
    mf_scaled = mf_scaled.loc[:, na_count_cols <= 0.20]  # Filtering out columns with high NA count

    # Checking which columns are na and deleting them for input data metafeatures
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
        cos_sim_ar[i] = cosine_similarity(comp_ar[i].reshape(1, -1), inp_ar)

    cos_sim_df = pd.DataFrame({"did": comp_ind, "Cosine Similarity": cos_sim_ar.reshape(-1)})
    cos_sim_df['dataset_name'] = extracted_mf[extracted_mf.index != openml_dataset].reset_index()['dataset_name']
    ranking = cos_sim_df.sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)
    return ranking


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dataset', help='Select did of dataset to base ranking on', type=int, default=14)
    parser.add_argument('--split',
                        help='Select metafeature extraction model (one can take the average of the metafeatures across all 5 splits)',
                        type=int, default=0)
    args = parser.parse_args()

    openml_dataset = args.input_dataset

    metafeatures_d2v = process_d2v(openml_dataset, split=args.split)

    metafeatures_mfe, mfe_extracted = process_mfe(openml_dataset)

    mfe_check = False
    if mfe_extracted:
        try:
            ranking = create_ranking(openml_dataset, "extracted_MF/OpenML-CC18_mfe.csv", metafeatures_mfe)
            mfe_check = True
        except:
            print("MFE ranking could not be computed.")

    ranking2 = create_ranking(openml_dataset, "extracted_MF/OpenML-CC18_d2v.csv", metafeatures_d2v)

    if mfe_check:
        print("Similarity ranking based on MFE metafeatures:")
        print(ranking.head(10))
        print("\n\n")

    print("Similarity ranking based on D2V metafeatures:")
    print(ranking2.head(10))


if __name__ == "__main__":
    main()
