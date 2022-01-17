from pymfe.mfe import MFE
import pandas as pd
from MFE.dataset_preprocessing import dataset_OpenML


def extract_features_OpenML(data_id):
    dat = dataset_OpenML(data_id)
    X, y = dat.get_arrays()
    print(dat.name + " has shape: " + str(X.shape))

    if X.shape[0] * X.shape[1] <= 1000000:
        mfe = MFE()
        mfe.fit(X, y)
        ft = mfe.extract(cat_cols='auto', suppress_warnings=True)
        return dat.name, pd.DataFrame(data=[ft[1]], index=[dat.name], columns=ft[0])

    else:
        print(dat.name + " is too large, MFE will not be computed.")
        return dat.name, None
