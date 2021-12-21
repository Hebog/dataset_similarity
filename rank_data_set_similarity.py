import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mf_path',
                    help='Specify path to CSV containing extracted metafeatures',
                    type=str, default="extracted/OpenML-CC18.csv")
parser.add_argument('--input_dataset', help='Select name of dataset to base ranking on', type=str, default="mfeat-fourier")
args = parser.parse_args()

extracted_mf = pd.read_csv(args.mf_path, index_col=0)
min_max_scaler = MinMaxScaler()
mf_scaled = min_max_scaler.fit_transform(extracted_mf)
mf_scaled = pd.DataFrame(mf_scaled, index=extracted_mf.index)

inp_ar = np.array(mf_scaled.loc[args.input_dataset,:]).reshape(1,-1)
filtered = mf_scaled[mf_scaled.index != args.input_dataset]
comp_ind, comp_ar = np.array(filtered.index), np.array(filtered)

cos_sim_ar = np.zeros((len(comp_ar), 1))
for i in range(len(comp_ar)):
    cos_sim_ar[i] = cosine_similarity(comp_ar[i].reshape(1,-1), inp_ar)

cos_sim_df = pd.DataFrame({"Dataset":comp_ind, "Cosine Similarity": cos_sim_ar.reshape(-1)})
ranking = cos_sim_df.sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)

print(ranking.head(10))