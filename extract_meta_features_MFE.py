import numpy as np
import os
import openml
import time
from MFE.extract_features import extract_features_OpenML



rootdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(rootdir, "extracted")
os.makedirs(save_dir, exist_ok=True)

suite = openml.study.get_suite(99)
save_path = os.path.join(save_dir, f"{suite.name[:11]}_mfe.csv")

time0 = time.time()

for openml_id in suite.data:
    print(openml_id)
    time1 = time.time()
    name, metafeatures = extract_features_OpenML(openml_id)
    metafeatures.to_csv(save_path, mode="a", header=not os.path.exists(save_path))

    time2 = time.time()

    print("Successfully finished dataset: " + name + "\nTook: " + str(np.round(time2 - time1, 2)) + " seconds")

print("Successfully finished all datasets, took: " + str(np.round(time2 - time0, 2)) + " seconds")