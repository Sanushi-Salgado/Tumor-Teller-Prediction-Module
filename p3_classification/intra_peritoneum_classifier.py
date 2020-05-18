# Author: Sanushi Salgado

import warnings

from imblearn.pipeline import make_pipeline
from sklearn import metrics

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, \
    VotingClassifier, StackingClassifier, IsolationForest
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_similarity_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# from evaluation.model_evaluation import print_evaluation_results
# from pre_processing.pre_processor import make_boolean, fill_missing_values, perform_one_hot_encoding
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







def intra_peritoneum_region_classifier():
    data_frame = pd.read_csv("../resources/datasets/intra_peritoneum_region.csv", na_values='?', dtype='category')
    data_frame.drop('region', axis=1, inplace=True)

    get_details(data_frame)

    print("Before Oversampling By Class\n", data_frame.groupby('class').size())
    # sns.countplot(data_frame['class'], label="Count")
    # plt.show()


    features = data_frame.drop(['class'], axis=1)
    labels = data_frame['class']  # Labels - 3, 5, 6, 7, 11, 12, 13

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

    smote = RandomOverSampler(sampling_strategy='minority')  # minority - hamming loss increases, accuracy, jaccard, avg f1, macro avg decreases
    X_resampled, y_resampled = smote.fit_sample(X_train, y_train)
    # X_resampled2, y_resampled2 = SMOTE().fit_resample(X_resampled2, y_resampled2)
    # pd.DataFrame(X_resampled).to_csv('resources/data/X_resampled2.csv', index=False)
    # pd.DataFrame(y_resampled).to_csv('resources/data/y_resampled2.csv', index=False)

    print("X_train2 ", pd.DataFrame(X_resampled).shape)
    print("X_train2 ", pd.DataFrame(y_resampled).shape)

    df = pd.DataFrame(y_resampled)
    print(df.groupby('class').size())


    sel_chi2 = SelectKBest(chi2, k=8)  # select 9 features
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




    # Spot Check Algorithms
    # spot_check_algorithms(X_train_chi2, y_resampled)




    # Make predictions on validation dataset using the selected model
    # intra_peritoneum_model =  KNeighborsClassifier(n_neighbors=5)  # kNN()- 0.39, kNN(neig-5) - 0.44, 0.39, LogisticRegression(solver='liblinear', multi_class='ovr'), wid 8, 9 features - 0.42, wid 12 featues - 0.40,  SVC(gamma='auto') - 0.35, OneVsRestClassifier(GaussianNB()) - 0.05

    intra_peritoneum_model = IsolationForest(n_estimators=100)
    #  Train the final model
    intra_peritoneum_model = intra_peritoneum_model.fit(X_train_chi2, y_resampled)

    # Evaluate the final model on the training set
    predictions = intra_peritoneum_model.predict(X_train_chi2)
    print_evaluation_results(y_resampled, predictions)

    # Evaluate the final model on the test set
    predictions = intra_peritoneum_model.predict(X_test_chi2)
    print_evaluation_results(y_test, predictions, train=False)

    joblib.dump(intra_peritoneum_model, filename='../resources/models/sub_classifier_3.pkl')