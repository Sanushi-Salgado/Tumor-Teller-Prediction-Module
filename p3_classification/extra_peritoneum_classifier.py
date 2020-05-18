from sklearn import metrics

from sklearn.externals import joblib



# Author: Sanushi Salgado

import warnings

from imblearn.combine import SMOTEENN
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline

import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz

from p1_eda.eda import get_details
from p2_preprocessing.data_cleansing import perform_one_hot_encoding
from p3_classification.spot_check import spot_check_algorithms
from p5_evaluation.model_evaluation import print_evaluation_results

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)






# replace 2 value of each specified column with 0
def make_boolean(data_frame):
    cols_to_replace = ['sex', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver', 'brain', 'skin', 'neck',
                       'supraclavicular', 'axillar', 'mediastinum', 'abdominal']

    for col in cols_to_replace:
        data_frame[col].replace('2', '0', inplace=True)
    print(data_frame.head())







def extra_peritoneum_region_classifier():
    data_frame = pd.read_csv("../resources/datasets/extra_peritoneum_region.csv", na_values='?', dtype='category')
    data_frame.drop('region', axis=1, inplace=True)


    get_details(data_frame)

    print("Before Oversampling By Class\n", data_frame.groupby('class').size())
    # make_boolean(data_frame)
    # sns.countplot(data_frame['class'], label="Count")
    # plt.show()



    features = data_frame.drop(['class'], axis=1)
    labels = data_frame['class']  # Labels - 8, 14, 15, 16, 17, 18, 19, 20, 21

    # pca = decomposition.PCA(n_components=9)
    # pca.fit(features)
    # features = pca.transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, shuffle=True)
    # pd.DataFrame(X_train).to_csv('resources/data/X_train2.csv', index=False)
    # pd.DataFrame(X_test).to_csv('resources/data/X_test2.csv', index=False)
    # pd.DataFrame(y_train).to_csv('resources/data/y_train2.csv', index=False)
    # pd.DataFrame(y_test).to_csv('resources/data/y_test2.csv', index=False)

    print("X_train2 ", pd.DataFrame(X_train).shape)
    print("X_train2 ", pd.DataFrame(y_train).shape)

    # smote = RandomOverSampler()  # minority - hamming loss increases, accuracy, jaccard, avg f1, macro avg decreases
    # X_resampled2, y_resampled2 = smote.fit_sample(X_train2, y_train2)
    # # X_resampled2, y_resampled2 = SMOTE().fit_resample(X_resampled2, y_resampled2)
    # pd.DataFrame(X_resampled2).to_csv('resources/data/X_resampled2.csv', index=False)
    # pd.DataFrame(y_resampled2).to_csv('resources/data/y_resampled2.csv', index=False)

    X_resampled2 = None
    y_resampled2 = None
    smote = RandomOverSampler()

    # for i in range(4):
    #     X_resampled2, y_resampled2 = smote.fit_resample(X_train2, y_train2)
    #     X_train2 = X_resampled2
    #     y_train2 = y_resampled2
    #     print("X_train2 ", pd.DataFrame(X_resampled2).shape)
    #     print("X_train2 ", pd.DataFrame(y_resampled2).shape)


    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    # pd.DataFrame(X_resampled).to_csv('resources/data/X_resampled2.csv', index=False)
    # pd.DataFrame(y_resampled).to_csv('resources/data/y_resampled2.csv', index=False)

    # print("X_train2 ", pd.DataFrame(X_resampled2).shape)
    # print("X_train2 ", pd.DataFrame(y_resampled2).shape)

    df = pd.DataFrame(y_resampled)
    print(df.groupby('class').size())

    sel_chi2 = SelectKBest(chi2, k=8)  # select 8 features
    X_train_chi2 = sel_chi2.fit_transform(X_resampled, y_resampled)
    print(sel_chi2.get_support())

    X_test_chi2 = sel_chi2.transform(X_test)
    print(X_test.shape)
    print(X_test_chi2.shape)

    # # ###############################################################################
    # # #                               4. Scale data                                 #
    # # ###############################################################################
    # sc = StandardScaler()
    # X_resampled2 = sc.fit_transform(X_resampled2)
    # X_test2 = sc.transform(X_test2)


    estimators = [('rf', RandomForestClassifier(random_state=42)),
                  ('svr', make_pipeline(StandardScaler(), KNeighborsClassifier()))]
    # clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(multi_class='ovr'))
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(multi_class='ovr'))



    # Spot Check Algorithms
    spot_check_algorithms(X_train_chi2, y_resampled)

    # Make predictions on validation dataset using the selected model
    # model2 = OneVsRestClassifier(GaussianNB()) # 0.43
    # model2 = DecisionTreeClassifier() # 0.78, 0.86, 0.72
    # model2 = RandomForestClassifier() # 0.80, 0.74, 0.69, 0.65, 0.66
    # model2 = GradientBoostingClassifier()  #
    # model2 =  VotingClassifier(estimators=[('rf', RandomForestClassifier()), ('mlp', MLPClassifier()), (('NB', GaussianNB()))], voting='hard') # 0.51



    extra_peritoneum_model = OneVsRestClassifier(RandomForestClassifier()) # -0.48, wid 8 features - 0.48, 0.52, 0.53, 0.54, 0.57, wid * f - 0.51,  clf - 0.48, kNN - 0.49

    #  Train the final model
    extra_peritoneum_model = extra_peritoneum_model.fit(X_train_chi2, y_resampled)

    # Evaluate the final model on the training set
    predictions = extra_peritoneum_model.predict(X_train_chi2)
    print_evaluation_results(y_resampled, predictions)

    # Evaluate the final model on the test set
    predictions = extra_peritoneum_model.predict(X_test_chi2)
    print_evaluation_results(y_test, predictions, train=False)

    joblib.dump(extra_peritoneum_model, filename='../resources/models/sub_classifier_4.pkl')
