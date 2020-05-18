# Author: Sanushi Salgado

import warnings
from sklearn import metrics

import graphviz
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
# from evaluation.model_evaluation import print_evaluation_results
# from pre_processing.pre_processor import make_boolean, fill_missing_values, perform_one_hot_encoding
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_similarity_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, learning_curve, \
    cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from p1_eda.eda import check_duplicates
from p2_preprocessing.data_cleansing import impute_missing_values, perform_one_hot_encoding
from p3_classification.spot_check import spot_check_algorithms

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)





def print_evaluation_results(y_test, predictions):
    print()
    print("!!!!!!!!!!!!!!!!!!!!! EVALUATION RESULTS !!!!!!!!!!!!!!!!!!!!!!")
    print("Accuracy Score ", accuracy_score(y_test, predictions))
    print("Hamming Loss ", hamming_loss(y_test, predictions))
    print("Jaccard Similarity Score ", jaccard_similarity_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
    print(classification_report_imbalanced(y_test, predictions))
    print()







def make_datasets(dataframe):
    upper_region_instances = []
    thoracic_region_instances = []
    intra_peritoneum_instances = []
    extra_peritoneum_instances = []

    column = dataframe['class']
    length = len(column)

    for row in range(length):
        value = column[row]
        if ( value == '2' or value == '4' or value == '10' ):
            record = dataframe.values[row]
            upper_region_instances.append(record)
        elif( value == '1' or value == '22' ):
            record = dataframe.values[row]
            thoracic_region_instances.append(record)
        elif( value == '3' or value == '5' or value == '6' or value == '7' or value == '11' or value == '12' or
              value == '13' ):
            record = dataframe.values[row]
            intra_peritoneum_instances.append(record)
        elif( value == '8' or value == '14' or value == '15' or value == '16' or value == '17' or value == '18' or
                        value == '19' or value == '20' or value == '21' ):
            record = dataframe.values[row]
            extra_peritoneum_instances.append(record)


    upper_region_dataframe = pd.DataFrame(upper_region_instances)
    thoracic_region_dataframe = pd.DataFrame(thoracic_region_instances)
    intra_peritoneum_region_dataframe = pd.DataFrame(intra_peritoneum_instances)
    extra_peritoneum_region_dataframe = pd.DataFrame(extra_peritoneum_instances)

    header = ['age', 'sex', 'histologic-type', 'degree-of-diffe', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum',
              'liver', 'brain', 'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal', 'region', 'class']

    upper_region_dataframe.to_csv('../resources/datasets/upper_region.csv', header=header, index=False)
    thoracic_region_dataframe.to_csv('../resources/datasets/thoracic_region.csv', header=header, index=False)
    intra_peritoneum_region_dataframe.to_csv('../resources/datasets/intra_peritoneum_region.csv', header=header, index=False)
    extra_peritoneum_region_dataframe.to_csv('../resources/datasets/extra_peritoneum_region.csv', header=header, index=False)







def classify_by_region():
    data_frame = pd.read_csv("../resources/datasets/preprocessed-primary-tumor-with-region.csv", na_values='?', dtype='category')
    data_frame.drop('class', axis=1)
    print(data_frame.shape)
    print(data_frame.head(10))
    print("Before Oversampling By Region\n", data_frame.groupby('region').size())


    # One Hot Encoding
    columns_to_encode = ['sex', 'histologic-type', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver',
                         'brain', 'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal']
    data_frame = perform_one_hot_encoding(data_frame, columns_to_encode)




    X = data_frame.drop(['region'], axis=1)  # Features - age, sex drop
    y = data_frame['region']  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    # X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, shuffle=True)
    # pd.DataFrame(X_train).to_csv('resources/data/X_train.csv', index=False)
    # pd.DataFrame(X_validation).to_csv('resources/data/X_validation.csv', index=False)
    # pd.DataFrame(X_test).to_csv('resources/data/X_test.csv', index=False)
    #
    # pd.DataFrame(y_train).to_csv('resources/data/y_train.csv', index=False)
    # pd.DataFrame(y_validation).to_csv('resources/data/y_validation.csv', index=False)
    # pd.DataFrame(y_test).to_csv('resources/data/y_test.csv', index=False)


    spot_check_algorithms(X_train, y_train)



    sm = SMOTE()
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    pd.DataFrame(X_resampled).to_csv('../resources/datasets/X_resampled.csv', index=False)
    pd.DataFrame(y_resampled).to_csv('../resources/datasets/y_resampled.csv', header=['region'], index=False)
    print("After Oversampling By Region\n", (pd.DataFrame(y_resampled)).groupby('region').size())



    ###############################################################################
    #                               4. Scale data                                 #
    ###############################################################################
    # sc = StandardScaler()
    # X_resampled = sc.fit_transform(X_resampled)
    # X_test = sc.transform(X_test)



    # mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes = [100]*5, random_state=42))


    # parent classifier - classify items between upper region, thoracic region, intra peritoneum & extra peritoneum region
    parent_model = MLPClassifier() # MLP wid fs - 0.65, 0.69, 0.70,   GB - 0.67, without fs 0.62, 0.61,    DT - 0.58,   RF - 0.67,  multi_LR - wid fs 0.64 , voting - 0.60

    # Train the final model & evaluate predictions of the model on both training and test sets
    parent_model.fit(X_resampled, y_resampled)
    predictions = parent_model.predict(X_resampled)
    print_evaluation_results(y_resampled, predictions)

    predictions = parent_model.predict(X_test)
    print_evaluation_results(y_test, predictions)

    # probs = parent_model.predict_proba(X_test)
    # print("Prediction probabilities for Region\n", probs)
    joblib.dump(parent_model, filename='../resources/models/parent_classifier.pkl')












data_frame = pd.read_csv("../resources/datasets/primary-tumor-with-region.csv", na_values='?', dtype='category')


# Impute missing values
print("Before imputing\n", data_frame.isnull().sum())
data_frame = impute_missing_values(data_frame, 'most_frequent')
print("After imputing\n", data_frame.isnull().sum())


# Drop duplicate records if exist
is_duplicated = check_duplicates(data_frame)
if is_duplicated:
    data_frame.drop_duplicates(inplace=True)
    print("Dropped duplicate records. Size after dropping duplicates: ", data_frame.shape)



header = ['age', 'sex', 'histologic-type', 'degree-of-diffe', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum',
          'liver', 'brain', 'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal', 'region', 'class']

# Write pre processed data to a file
data_frame.to_csv("../resources/datasets/preprocessed-primary-tumor-with-region.csv", header=header, index=False)


# Load dataset
data_frame = pd.read_csv("../resources/datasets/preprocessed-primary-tumor-with-region.csv", dtype='category')
# Create separate datasets
make_datasets(data_frame)


##################################### Parent Classifier ##################################
classify_by_region()

##################################### Sub Classifier 01 ##################################
# upper_region_classifier()

##################################### Sub Classifier 02 ##################################
# thoracic_region_classifier()

##################################### Sub Classifier 03 ##################################
# intra_peritoneum_region_classifier()

##################################### Sub Classifier 04 ##################################
# extra_peritoneum_region_classifier()








