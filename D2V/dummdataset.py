#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:41:34 2021

@author: hsjomaa
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import openml
np.random.seed(92)

def ptp(X):
    param_nos_to_extract = X.shape[1]
    domain = np.zeros((param_nos_to_extract, 2))
    for i in range(param_nos_to_extract):
        domain[i, 0] = np.min(X[:, i])
        domain[i, 1] = np.max(X[:, i])
    X = (X - domain[:, 0]) / np.ptp(domain, axis=1)
    return X

def flatten(x,y):
    '''
    Genearte x_i,y_j for all i,j \in |x|

    Parameters
    ----------
    x : numpy.array
        predictors; shape = (n,m)
    y : numpy.array
        targets; shape = (n,t)

    Returns
    -------
    numpy.array()
        shape ((n\times m\times t)\times 2).

    '''
    x_stack = []
    for c in range(y.shape[1]):
        c_label = np.tile(y[:,c],reps=[x.shape[1]]).transpose().reshape(-1,1)
        x_stack.append(np.concatenate([x.transpose().reshape(-1,1),c_label],axis=-1))
    return np.vstack(x_stack)

class Dataset(object):
    
    def __init__(self,file,rootdir):
            # read dataset
            self.X,self.y = self.__get_data(file,rootdir=rootdir)

            # batch properties
            self.ninstanc = 256
            self.nclasses = 3
            self.nfeature = 16
            
             
    def __get_data(self,file,rootdir):

        # read dataset folds
        datadir = os.path.join(rootdir, "datasets", file)
        # read internal predictors
        data = pd.read_csv(f"{datadir}/{file}_py.dat",header=None)
        # transform to numpy
        data    = np.asarray(data)
        # read internal target
        labels = pd.read_csv(f"{datadir}/labels_py.dat",header=None)
        # transform to numpy
        labels    = np.asarray(labels)        

        return data,labels

    def sample_batch(self,data,labels,ninstanc,nclasses,nfeature):
        '''
        Sample a batch from the dataset of size (ninstanc,nfeature)
        and a corresponding label of shape (ninstanc,nclasses).

        Parameters
        ----------
        data : numpy.array
            dataset; shape (N,F) with N >= nisntanc and F >= nfeature
        labels : numpy.array
            categorical labels; shape (N,) with N >= nisntanc
        ninstanc : int
            Number of instances in the output batch.
        nclasses : int
            Number of classes in the output label.
        nfeature : int
            Number of features in the output batch.

        Returns
        -------
        data : numpy.array
            subset of the original dataset.
        labels : numpy.array
            one-hot encoded label representation of the classes in the subset

        '''
        # Create the one-hot encoder
        ohc           = OneHotEncoder(categories = [range(len(np.unique(labels)))],sparse=False)
        d = {ni: indi for indi, ni in enumerate(np.unique(labels))}
        # process the labels
        labels        = np.asarray([d[ni] for ni in labels.reshape(-1)]).reshape(-1)
        # transform the labels to one-hot encoding
        labels        = ohc.fit_transform(labels.reshape(-1,1))
        # ninstance should be less than or equal to the dataset size
        ninstanc            = np.random.choice(np.arange(0,data.shape[0]),size=np.minimum(ninstanc,data.shape[0]),replace=False)
        # nfeature should be less than or equal to the dataset size
        nfeature         = np.random.choice(np.arange(0,data.shape[1]),size=np.minimum(nfeature,data.shape[1]),replace=False)
        # nclasses should be less than or equal to the total number of labels
        nclasses         = np.random.choice(np.arange(0,labels.shape[1]),size=np.minimum(nclasses,labels.shape[1]),replace=False)
        # extract data at selected instances
        data          = data[ninstanc]
        # extract labels at selected instances
        labels        = labels[ninstanc]
        # extract selected features from the data
        data          = data[:,nfeature]
        # extract selected labels from the data
        labels        = labels[:,nclasses]
        return data,labels
    
    def instances(self,ninstanc=None,nclasses=None,nfeature=None):
        # check if ninstance is provided
        ninstanc = ninstanc if ninstanc is not None else self.ninstanc
        # check if ninstance is provided
        nclasses = nclasses if nclasses is not None else self.nclasses
        # check if ninstance is provided
        nfeature = nfeature if nfeature is not None else self.nfeature        
        # check if neg batch is provided
        instance_x,instance_i = [],[]
        # append information to the placeholders
        x,y = self.sample_batch(self.X,self.y,ninstanc,nclasses,nfeature)
        instance_i.append(x.shape+(y.shape[1],)+(-1,))
        instance_x.append(flatten(x,y))
        # remove x,y
        del x,y
        # stack x values
        x = np.vstack(instance_x)
        # stack ninstanc
        ninstance = np.vstack(instance_i)[:,0][:,None]
        # stack nfeatures
        nfeature = np.vstack(instance_i)[:,1][:,None]
        # stack nclasses
        nclasses = np.vstack(instance_i)[:,2][:,None]
        # get task description of surr task
        return x,ninstance,nfeature,nclasses


class Dataset_OpenML(Dataset):
    """
    Create an OpenML specific version of the dataset class.
    """
    def __init__(self, data_id):
        # read dataset
        self.X, self.y, self.name, self.num_target = self.__get_data(data_id)

        # batch properties
        self.ninstanc = 256
        self.nclasses = 3
        self.nfeature = 16

    def __get_data(self, data_id):
        # Retrieve dataset from OpenMl
        dataset = openml.datasets.get_dataset(data_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        data = self.preprocess_features(X, categorical_indicator)

        numerical_target = is_numeric_dtype(y)

        if numerical_target:
            # Scale labels to range from [0,1]
            labels = self.scale_labels(y)

        else:
            # Transform labels into categorical encoding
            labels = self.encode_labels(y)

        return data, labels, dataset.name, numerical_target

    def preprocess_features(self, X, categorical_indicator):
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


    def scale_labels(self, y):
        min_max_scaler = MinMaxScaler()
        return min_max_scaler.fit_transform(y)


    def encode_labels(self, y):
        le = LabelEncoder()
        return np.asarray(le.fit_transform(y))


    def sample_batch(self, data, labels, ninstanc, nclasses, nfeature):
        '''
        Sample a batch from the dataset of size (ninstanc,nfeature)
        and a corresponding label of shape (ninstanc,nclasses).

        Parameters
        ----------
        data : numpy.array
            dataset; shape (N,F) with N >= nisntanc and F >= nfeature
        labels : numpy.array
            categorical labels; shape (N,) with N >= nisntanc
        ninstanc : int
            Number of instances in the output batch.
        nclasses : int
            Number of classes in the output label.
        nfeature : int
            Number of features in the output batch.

        Returns
        -------
        data : numpy.array
            subset of the original dataset.
        labels : numpy.array
            one-hot encoded label representation of the classes in the subset

        '''
        # Create the one-hot encoder
        ohc = OneHotEncoder(categories=[range(len(np.unique(labels)))], sparse=False)
        d = {ni: indi for indi, ni in enumerate(np.unique(labels))}
        # process the labels
        labels = np.asarray([d[ni] for ni in labels.reshape(-1)]).reshape(-1)
        # transform the labels to one-hot encoding
        labels = ohc.fit_transform(labels.reshape(-1, 1))
        # ninstance should be less than or equal to the dataset size
        ninstanc = np.random.choice(np.arange(0, data.shape[0]), size=np.minimum(ninstanc, data.shape[0]),
                                    replace=False)
        # nfeature should be less than or equal to the dataset size
        nfeature = np.random.choice(np.arange(0, data.shape[1]), size=np.minimum(nfeature, data.shape[1]),
                                    replace=False)
        # nclasses should be less than or equal to the total number of labels
        nclasses = np.random.choice(np.arange(0, labels.shape[1]), size=np.minimum(nclasses, labels.shape[1]),
                                    replace=False)
        # extract data at selected instances
        data = data[ninstanc]
        # extract labels at selected instances
        labels = labels[ninstanc]
        # extract selected features from the data
        data = data[:, nfeature]
        # extract selected labels from the data
        labels = labels[:, nclasses]
        # print("shape data: " + str(data.shape))
        # print("shape labels: " + str(labels.shape))
        return data, labels

    def instances(self, ninstanc=None, nclasses=None, nfeature=None):
        # check if ninstance is provided
        ninstanc = ninstanc if ninstanc is not None else self.ninstanc
        # check if ninstance is provided
        nclasses = nclasses if nclasses is not None else self.nclasses
        # check if ninstance is provided
        nfeature = nfeature if nfeature is not None else self.nfeature
        # check if neg batch is provided
        instance_x, instance_i = [], []
        # append information to the placeholders
        x, y = self.sample_batch(self.X, self.y, ninstanc, nclasses, nfeature)
        instance_i.append(x.shape + (y.shape[1],) + (-1,))
        instance_x.append(flatten(x, y))
        # remove x,y
        del x, y
        # stack x values
        x = np.vstack(instance_x)
        # stack ninstanc
        ninstance = np.vstack(instance_i)[:, 0][:, None]
        # stack nfeatures
        nfeature = np.vstack(instance_i)[:, 1][:, None]
        # stack nclasses
        nclasses = np.vstack(instance_i)[:, 2][:, None]
        # get task description of surr task
        return x, ninstance, nfeature, nclasses
        
