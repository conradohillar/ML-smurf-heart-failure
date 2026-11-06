import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression as mutual_info
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')


def load_data():
    X_train = pd.read_csv('data/data_student/data_labeled/X_train.csv')
    X_test = pd.read_csv('data/data_student/data_labeled/X_test.csv')
    X_unlabeled = pd.read_csv('data/data_student/data_unlabeled/X.csv')
    y_train = pd.read_csv('data/data_student/data_labeled/y_train.csv', header=None)
    y_test = pd.read_csv('data/data_student/data_labeled/y_test.csv', header=None)

    return X_train, X_test, X_unlabeled, y_train, y_test

def preprocess_data(X_train, X_test, X_unlabeled):
    datasets = [X_train, X_test, X_unlabeled]
    
    drop_columns(datasets, ["img_filename", "profession"])

    ordinal_columns = ["sarsaparilla", "smurfberry_liquor", "smurfin_donuts"]

    ordinal_encoder = OrdinalEncoder(
        categories=[["very_low", "low", "moderate", "high", "very_high"]] * len(ordinal_columns)
    )

    X_train[ordinal_columns] = ordinal_encoder.fit_transform(X_train[ordinal_columns])
    X_test[ordinal_columns] = ordinal_encoder.transform(X_test[ordinal_columns])
    X_unlabeled[ordinal_columns] = ordinal_encoder.transform(X_unlabeled[ordinal_columns])

    onehot_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    profession_train = onehot_encoder.fit_transform(X_train[["profession"]])
    profession_test = onehot_encoder.transform(X_test[["profession"]])
    profession_unlabeled = onehot_encoder.transform(X_unlabeled[["profession"]])

    profession_columns = onehot_encoder.get_feature_names_out(["profession"])

    X_train = X_train.drop(columns=["profession"])
    X_test = X_test.drop(columns=["profession"])
    X_unlabeled = X_unlabeled.drop(columns=["profession"])

    X_train[profession_columns] = profession_train
    X_test[profession_columns] = profession_test
    X_unlabeled[profession_columns] = profession_unlabeled

def drop_columns(datasets, columns_to_drop):
    for dataset in datasets:
        dataset.drop(columns=columns_to_drop, inplace=True)

def main():
    X_train, X_test, X_unlabeled, y_train, y_test = load_data()
    print(X_train.head(5))
    preprocess_data(X_train, X_test, X_unlabeled)
    print("Data loaded and preprocessed successfully.")

if __name__ == "__main__":
    main()