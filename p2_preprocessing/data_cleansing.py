from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd



def impute_missing_values(data_frame, strategy):
    columns = data_frame.columns
    imp = SimpleImputer( missing_values=np.nan, strategy=strategy, fill_value=None, verbose=0, copy=True )
    imputed = imp.fit_transform(data_frame)
    data_frame = pd.DataFrame(imputed, columns=columns)
    print("After imputing\n", data_frame.isnull().sum().sum())
    print(data_frame.head())
    return data_frame




def perform_one_hot_encoding(data_frame, columns_to_encode):
    data_frame = pd.get_dummies( data=data_frame, columns=columns_to_encode, drop_first=True )
    print("After OHE\n", data_frame.head())
    print(data_frame.info())
    return data_frame


# def perform_label_encoding(data_frame):


