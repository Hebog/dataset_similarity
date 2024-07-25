import importlib
import os
import warnings

import numpy as np
import openml
import pandas as pd
import sklearn
from openml.tasks import TaskType
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


from rank_data_set_similarity import create_ranking#, process_d2v, process_mfe


def get_task_df(did, task_type="S_Classification"):
    if task_type == "S_Classification":
        tt = TaskType.SUPERVISED_CLASSIFICATION
    elif task_type == "S_Regression":
        tt = TaskType.SUPERVISED_REGRESSION

    task_df = openml.tasks.list_tasks(data_id=did, task_type=tt,
                                      output_format='dataframe')
    # task_df_filtered = task_df.query("estimation_procedure=='10-fold Crossvalidation' and evaluation_measures=='predictive_accuracy'")

    return task_df.reset_index(drop=True)


def get_best_run(did):
    task_df = get_task_df(did, task_type='S_Classification')
    task_ids = task_df['tid'].to_numpy()
    evaluation_df = get_evaluations(task_ids)
    freq_task_id = get_most_frequent_task(evaluation_df)
    evaluation_df_freq_id = get_evaluations([freq_task_id])
    best_flow_id = evaluation_df_freq_id.loc[0, 'flow_id']
    best_run_id = evaluation_df_freq_id.loc[0, 'run_id']
    best_run = openml.runs.get_run(best_run_id)
    return best_run


def get_second_best_run(did):
    task_df = get_task_df(did, task_type='S_Classification')
    task_ids = task_df['tid'].to_numpy()
    evaluation_df = get_evaluations(task_ids)
    freq_task_id = get_most_frequent_task(evaluation_df)
    evaluation_df_freq_id = get_evaluations([freq_task_id])
    best_flow_id = evaluation_df_freq_id.loc[0, 'flow_id']
    for i in range(len(evaluation_df_freq_id)):
        best_run_id = evaluation_df_freq_id.loc[i, 'run_id']
        best_run = openml.runs.get_run(best_run_id)
        if ('KerasClassifier' not in best_run.flow_name) and ('sklearn_extra' not in best_run.flow_name):
            return best_run


def get_evaluations(task_ids):
    eval_df = openml.evaluations.list_evaluations(function='predictive_accuracy', tasks=task_ids,
                                                  output_format='dataframe', sort_order="desc")
    eval_df_filtered = eval_df[eval_df['flow_name'].str.contains("sklearn")]
    return eval_df_filtered.reset_index(drop=True)


def get_most_frequent_task(evaluation_df):
    return evaluation_df['task_id'].value_counts().idxmax()


def get_model(
        model_name: str, import_module: str, model_params: dict
) -> sklearn.base.BaseEstimator:
    """Returns a scikit-learn model."""
    model_class = getattr(importlib.import_module(import_module), model_name)
    model = model_class(**model_params)  # Instantiates the model
    return model


def get_classifier_run(did):
    run = get_best_run(did)
    if "KerasClassifier" in run.flow_name:
        print("Getting worse run because of keras classifier")
        run = get_second_best_run(did)
    if "sklearn_extra" in run.flow_name:
        print("Getting worse run because of sklearn_extra classifier")
        run = get_second_best_run(did)

    exceptions = {'SVC': 'sklearn.svm', 'AdaBoostClassifier': 'sklearn.ensemble',
                  'MLPClassifier': 'sklearn.neural_network',
                  'RandomForestClassifier': 'sklearn.ensemble', 'ExtraTreesClassifier': 'sklearn.ensemble'}
    flow_name = run.flow_name
    print("getting classifier for:" + str(did))
    #     print(flow_name)

    classifier_str = flow_name.split(",")[-1]
    classifier_str = classifier_str.split("=")[1]
    classifier_str = classifier_str.split(")")[0]

    if "(" in classifier_str:
        classifier_str = classifier_str.split("(")[0]

    split_str = classifier_str.split('.')
    model_str = split_str[-1]
    module_str = '.'.join(split_str[:-1])

    print('model string: ' + model_str)
    print('module string: ' + module_str)

    if not model_str in exceptions.keys():
        model_instance = get_model(model_str, module_str, {})

    else:
        model_instance = get_model(model_str, exceptions[model_str], {})

    return model_instance


def reconstruct_pipeline(run):
    if not run.flow_name.startswith("sklearn.pipeline.Pipeline("):
        warnings.warn("Warning: run does not contain sklearn pipeline, pipeline could not be reconstructed.")
        return None
    print(run.flow_name)
    pipeline_steps_str = run.flow_name.split("sklearn.pipeline.Pipeline(")[1].strip('()1').split(',')
    pipeline_steps = []
    for step in pipeline_steps_str:
        class_name, class_str = step.split("=")[0], step.split("=")[1]
        split_str = class_str.split('.')
        model_str = split_str[-1]
        module_str = '.'.join(split_str[:-1])
        if module_str.startswith("sklearn"):
            class_instance = get_model(model_str, module_str, {})
            pipeline_steps.append((class_name, class_instance))

        else:
            print("skipped: " + module_str + model_str)

    return Pipeline(pipeline_steps)


def impute_missing(X, categorical_indicator):
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


def scale_labels(y):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(y)


def encode_labels(y):
    le = LabelEncoder()
    return np.asarray(le.fit_transform(y))


def preprocess_features(X, categorical_indicator):
    '''
    Preprocess the feature table X by imputing categorical and numeric features with constant value and mean
    respectively, by encoding the categorical features from 1..N_categories and by standardizing the features
    to have mean=0 and std=1.

    :param X: pandas DataFrame object contain feature table
    :param categorical_indicator: list of booleans indicating whether columns are categorical
    :return: numpy array with preprocessed features
    '''
    numeric_features = X.columns[~np.array(categorical_indicator)]
    numeric_transformer = SimpleImputer(strategy="mean")

    categorical_features = X.columns[categorical_indicator]
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
               ("ordinal_encoder", OrdinalEncoder())]
    )

    type_specific_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    preprocessor = Pipeline(
        steps=[("type_preprocessor", type_specific_preprocessor), ("scaler", StandardScaler())]
    )

    return preprocessor.fit_transform(X)


def get_most_similar_did(did, ranking_type):
    if ranking_type == "MFE":
        # metafeatures_mfe, mfe_extracted = process_mfe(did)
        metafeatures_mfe = None
        mfe_extracted = True
        mfe_check = False
        if mfe_extracted:
            try:
                ranking = create_ranking(did, "dataset_similarity/extracted_MF/OpenML-CC18_mfe.csv", metafeatures_mfe)
                mfe_check = True

                return ranking.loc[0, 'did'], mfe_check
            except:
                print("MFE ranking could not be computed.")

                return None, mfe_check

        else:
            return None, mfe_check

    if ranking_type == "PFN":
        # metafeatures_pfn, pfn_extracted = process_pfn(did)
        metafeatures_pfn = None
        pfn_check = False
        pfn_extracted =True
        if pfn_extracted:
            try:
                ranking = create_ranking(did, "dataset_similarity/extracted_MF/OpenML-CC18_pfn.csv", metafeatures_pfn)
                pfn_check = True

                return ranking.loc[0, 'did'], pfn_check
            except:
                print("PFN ranking could not be computed.")

                return None, pfn_check

        else:
            return None, pfn_check
    
    if ranking_type == "D2V":
        # metafeatures_d2v = process_d2v(did, split=0)
        metafeatures_d2v = None
        ranking = create_ranking(did, "dataset_similarity/extracted_MF/OpenML-CC18_d2v.csv", metafeatures_d2v)
        return ranking.loc[0, 'did']


def evaluate_ranking(did):
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    numerical_target = is_numeric_dtype(y)

    did_most_similar_d2v = get_most_similar_did(did, "D2V")
    print("d2v most similar did: " + str(did_most_similar_d2v))
    did_most_similar_mfe, mfe_success = get_most_similar_did(did, "MFE")

    if mfe_success:
        print("mfe most similar did: " + str(did_most_similar_mfe))

    did_most_similar_pfn, pfn_success = get_most_similar_did(did, "PFN")

    if pfn_success:
        print("pfn most similar did: " + str(did_most_similar_pfn))

    dc_lst = []
    dt_lst = []
    bc_d2v_lst = []
    bc_mfe_lst = []
    bc_pfn_lst = []
    
    kf = KFold(n_splits=5, shuffle=True)
    for train, test in kf.split(X):
        X_train, y_train = X.iloc[train], y[train]
        X_test, y_test = X.iloc[test], y[test]

        X_train = preprocess_features(X_train, categorical_indicator)
        X_test = preprocess_features(X_test, categorical_indicator)

        if numerical_target:
            # Scale labels to range from [0,1]
            y_train = scale_labels(y_train)
            y_test = scale_labels(y_test)

        else:
            # Transform labels into categorical encoding
            y_train = encode_labels(y_train)
            y_test = encode_labels(y_test)

        # y = np.asarray(y)

        dc = DummyClassifier(strategy="most_frequent")
        dc.fit(X_train, y_train)

        dt = DecisionTreeClassifier(max_depth=5)
        dt.fit(X_train, y_train)

        # D2V most similar best classifier
        bc_d2v = get_classifier_run(did_most_similar_d2v)
        bc_d2v.fit(X_train, y_train)

        dc_lst.append(accuracy_score(y_test, dc.predict(X_test)))
        dt_lst.append(accuracy_score(y_test, dt.predict(X_test)))
        bc_d2v_lst.append(accuracy_score(y_test, bc_d2v.predict(X_test)))

        # MFE most similar best classifier
        if mfe_success:
            bc_mfe = get_classifier_run(did_most_similar_mfe)
            bc_mfe.fit(X_train, y_train)
            bc_mfe_lst.append(accuracy_score(y_test, bc_mfe.predict(X_test)))

        else:
            bc_mfe_lst.append(np.nan)
            
        # PFN most similar best classifier
        bc_pfn = get_classifier_run(did_most_similar_pfn)
        bc_pfn.fit(X_train, y_train)
        bc_pfn_lst.append(accuracy_score(y_test, bc_pfn.predict(X_test)))
        

    return (dataset.name, np.mean(dc_lst), np.mean(dt_lst), np.mean(bc_d2v_lst), np.mean(bc_mfe_lst), np.mean(bc_pfn_lst))

def eval_portfolio(did):
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    numerical_target = is_numeric_dtype(y)
    portfolio = [KNeighborsClassifier, RidgeClassifier, SVC, DecisionTreeClassifier, GaussianNB,]
    accuracies = {algo.__name__: [] for algo in portfolio}
    for algo in portfolio:
        kf = KFold(n_splits=5, shuffle=True)
        for train, test in kf.split(X):
            X_train, y_train = X.iloc[train], y[train]
            X_test, y_test = X.iloc[test], y[test]

            X_train = preprocess_features(X_train, categorical_indicator)
            X_test = preprocess_features(X_test, categorical_indicator)

            if numerical_target:
                # Scale labels to range from [0,1]
                y_train = scale_labels(y_train)
                y_test = scale_labels(y_test)

            else:
                # Transform labels into categorical encoding
                y_train = encode_labels(y_train)
                y_test = encode_labels(y_test)

            # y = np.asarray(y)

            cl = algo()
            cl.fit(X_train, y_train)

            accuracies[algo.__name__].append(accuracy_score(y_test, cl.predict(X_test)))
    
    accuracies = {k: [np.mean(v)] for k,v in accuracies.items()}
    accuracies['name'] = dataset.name
    accuracies["data_id"] = did
    return accuracies


def main():
    save_path = "dataset_similarity/similarity_evaluation/similarity_evaluation_pfn.csv"

    for data_id in tqdm((28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 458, 469, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 1510, 1489, 1494, 1497, 1480, 1487, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 23381, 40668, 40966, 40982, 40994, 40983, )):#openml.study.get_suite(99).data[24:]):
        print("starting: " + str(data_id))
        try:
            if not data_id in []:
                evaluated_ranking = evaluate_ranking(data_id)
                evaluated_df = pd.DataFrame({"data_id": [data_id], 'name': [evaluated_ranking[0]],
                                            'dummy_clf': [evaluated_ranking[1]], 'dt_clf': [evaluated_ranking[2]],
                                            'd2v_clf': [evaluated_ranking[3]], "mfe_clf": [evaluated_ranking[4]],
                                            "pfn_clf": [evaluated_ranking[5]]})
                evaluated_df.to_csv(save_path, mode="a", header=not os.path.exists(save_path))
                print("processed: " + str(data_id))
        except Exception as e :
            print(e)

from multiprocessing import Manager
from multiprocessing import Pool  

def worker(data_id, extracted_mf, res):
        if data_id in extracted_mf.index:
            print("Input index already in features")
        else:
            try:
                if not data_id in []:
                    print("start: " + str(data_id))
                    evaluated_portfolio = eval_portfolio(data_id)
                    evaluated_df = pd.DataFrame(evaluated_portfolio)
                    print("processed: " + str(data_id))
                    res.append(evaluated_df)
            except Exception as e :
                print(e)


DIDS = (3,11, 14, 15, 16, 18, 22, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 458, 469, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 1510, 1489, 1494, 1497, 1480, 1487, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 23381,40668, 40966, 40982, 40994, 40983, )#
# DIDS = (3,11, 14, 15, )

def main1():
    save_path = "dataset_similarity/similarity_evaluation/similarity_evaluation_portfolio1.csv"
    pool_parameter = DIDS # list of objects to process
    try:
        extracted_mf = pd.read_csv(save_path, index_col="data_id")
    except FileNotFoundError:
        extracted_mf = pd.DataFrame()
    with Manager() as mgr:
        res = mgr.list([])
        params = []
        # build list of parameters to send to starmap
        for param in pool_parameter:
            params.append([param,extracted_mf, res])

        with Pool() as p:
            p.starmap(worker, tqdm(params, total=len(pool_parameter)) )
        print(res)
        

        print(res)
        evaluated_df = pd.concat(res)
        evaluated_df.to_csv(save_path, mode="a", header=not os.path.exists(save_path))


if __name__ == "__main__":
    main1()
