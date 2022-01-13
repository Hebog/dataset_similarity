import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import openml


class dataset_OpenML(object):
    """
    Create an OpenML specific version of the dataset class.
    """

    def __init__(self, data_id):
        # read dataset
        self.X, self.y, self.name = self.__get_data(data_id)

    def __get_data(self, data_id):
        # Retrieve dataset from OpenMl
        dataset = openml.datasets.get_dataset(data_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = self.impute_missing(X, categorical_indicator)

        return X, y, dataset.name

    def impute_missing(self, X, categorical_indicator):
        categorical_features = list(X.columns[categorical_indicator])
        numeric_features = list(X.columns[~np.array(categorical_indicator)])
        X = X.dropna(axis=1, how='all')

        categorical_features = [ft for ft in categorical_features if ft in X.columns]
        numeric_features = [ft for ft in numeric_features if ft in X.columns]

        object_cols = {col: 'object' for col in categorical_features}
        X = X.astype(object_cols)

        dtypes = dict(X.dtypes)

        numeric_transformer = SimpleImputer(strategy="mean")
        categorical_transformer = SimpleImputer(strategy="constant", fill_value="missing")
        type_specific_preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        prepr_np = type_specific_preprocessor.fit_transform(X)
        prepr_df = pd.DataFrame(prepr_np, columns=numeric_features + categorical_features)
        prepr_df = prepr_df.astype(dtypes)
        return prepr_df

    def get_arrays(self):
        return np.asarray(self.X), np.asarray(self.y)
