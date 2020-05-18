# Author: Sanushi Salgado

import warnings

from boruta import BorutaPy
from imblearn.under_sampling import ClusterCentroids
from sklearn import metrics

import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler, KMeansSMOTE, SMOTE, BorderlineSMOTE, ADASYN, SMOTENC, SVMSMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, \
    VotingClassifier, StackingClassifier
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_similarity_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold, \
    cross_val_predict, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz

from p1_eda.eda import get_details
from p2_preprocessing.data_cleansing import perform_one_hot_encoding
from p3_classification.baseline import get_baseline_performance
# from p3_classification.spot_check import spot_check_algorithms
from p5_evaluation.model_evaluation import print_evaluation_results

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)





############################################## STAGE 2 #############################################
# Then train a classifier with items of parent category 1

def upper_region_classifier():
    # Read in data
    data_frame = pd.read_csv("../resources/datasets/upper_region.csv", na_values='?', dtype='category')
    data_frame.drop('region', axis=1, inplace=True)

    get_details(data_frame)

    print("Before Oversampling By Class\n", data_frame.groupby('class').size())
    # sns.countplot(data_frame['class'], label="Count")
    # plt.show()


    features = data_frame.drop(['class'], axis=1)
    labels = data_frame['class']  # Labels - 2, 4, 10

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, shuffle=True)


    print("X_train2 ", pd.DataFrame(X_train).shape)
    print("X_train2 ", pd.DataFrame(y_train).shape)


    ros = RandomOverSampler()  # minority - hamming loss increases, accuracy, jaccard, avg f1, macro avg decreases
    X_resampled, y_resampled = ros.fit_sample(X_train, y_train)

    print("X_train2 ", pd.DataFrame(X_resampled).shape)
    print("X_train2 ", pd.DataFrame(y_resampled).shape)

    df = pd.DataFrame(y_resampled)
    print(df.groupby('class').size())


    # https://datascienceplus.com/selecting-categorical-features-in-customer-attrition-prediction-using-python/
    # categorical feature selection
    # sf = SelectKBest(chi2, k='all')
    # sf_fit = sf.fit(X_train, y_train)
    # # print feature scores
    # for i in range(len(sf_fit.scores_)):
    #     print(' %s: %f' % (X_train.columns[i], sf_fit.scores_[i]))
    #
    # # plot the scores
    # datset = pd.DataFrame()
    # datset['feature'] = X_train.columns[range(len(sf_fit.scores_))]
    # datset['scores'] = sf_fit.scores_
    # datset = datset.sort_values(by='scores', ascending=True)
    # sns.barplot(datset['scores'], datset['feature'], color='blue')
    # sns.set_style('whitegrid')
    # plt.ylabel('Categorical Feature', fontsize=18)
    # plt.xlabel('Score', fontsize=18)
    # plt.show()

    # sel_chi2 = SelectKBest(chi2, k=9)  # DT chi 9- 0.83*2, 10- 91,92
    #                                     # RF chi 9- 1,1,91,1,91
    #                                     # GB chi 9- 91,91,82,91
    # X_train_chi2 = sel_chi2.fit_transform(X_resampled, y_resampled)
    # X_test_chi2 = sel_chi2.transform(X_test)

    # # ###############################################################################
    # # #                               4. Scale data                                 #
    # # ###############################################################################
    # sc = StandardScaler()
    # X_resampled2 = sc.fit_transform(X_resampled2)
    # X_test2 = sc.transform(X_test2)


    # get_baseline_performance(X_resampled, y_resampled, X_test, y_test)

    # Spot Check Algorithms
    # spot_check_algorithms(X_resampled, y_resampled)




    # Make predictions on validation dataset using the selected model
    # upper_region_model = MLP # DT()- 0.92*4,   RF()-0.91,   RF(n_estimators=200) - 0.91, 1.0, # kNN - 0.91, knn(neigh-3) - 0.78, knn(neigh-5) - 0.91, DT - 0.79, 0.91, 0.92,  SVC(gamma='auto') - 0.91, LogisticRegression(solver='liblinear', multi_class='ovr') - 0.85,


    upper_region_model = RandomForestClassifier(n_jobs=-1,  max_depth=20, n_estimators=200)

    # define Boruta feature selection method
    # feat_selector = BorutaPy(upper_region_model, n_estimators='auto', verbose=2, random_state=1)
    # # find all relevant features - 5 features should be selected
    # feat_selector.fit(X_resampled, y_resampled)
    # # check selected features - first 5 features are selected
    # print(feat_selector.support_)
    # # check ranking of features
    # print(feat_selector.ranking_)
    # # call transform() on X to filter it down to selected features
    # X_filtered = feat_selector.transform(X_resampled)

    # Train the final model
    upper_region_model = upper_region_model.fit(X_resampled, y_resampled)

    # Evaluate the final model on the training set
    predictions = upper_region_model.predict(X_resampled)
    print_evaluation_results(y_resampled, predictions)

    # Evaluate the final model on the test set
    predictions = upper_region_model.predict(X_test)
    print_evaluation_results(y_test, predictions, train=False)

    # joblib.dump(upper_region_model, filename='../resources/models/sub_classifier_1.pkl')


