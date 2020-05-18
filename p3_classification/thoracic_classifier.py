# Author: Sanushi Salgado

import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, \
    VotingClassifier, ExtraTreesClassifier
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_similarity_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# from evaluation.model_evaluation import print_evaluation_results
# from pre_processing.pre_processor import make_boolean, fill_missing_values, perform_one_hot_encoding
# from a import make_boolean
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





def thoracic_region_classifier():
    data_frame = pd.read_csv("../resources/datasets/thoracic_region.csv", na_values='?', dtype='category')
    data_frame.drop('region', axis=1, inplace=True)

    get_details(data_frame)

    # make_boolean(data_frame)
    print("Before Oversampling By Class\n", data_frame.groupby('class').size())
    # sns.countplot(data_frame['class'], label="Count")
    # plt.show()


    features = data_frame.drop(['class'], axis=1)
    labels = data_frame['class']  # Labels - 1, 22


    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, shuffle=True)
    # pd.DataFrame(X_train).to_csv('resources/data/X_train2.csv', index=False)
    # pd.DataFrame(X_test).to_csv('resources/data/X_test2.csv', index=False)
    # pd.DataFrame(y_train).to_csv('resources/data/y_train2.csv', index=False)
    # pd.DataFrame(y_test).to_csv('resources/data/y_test2.csv', index=False)

    print("X_train2 ", pd.DataFrame(X_train).shape)
    print("X_train2 ", pd.DataFrame(y_train).shape)

    # smote = BorderlineSMOTE()
    smote = RandomOverSampler()  # minority - hamming loss increases, accuracy, jaccard, avg f1, macro avg decreases
    X_resampled2, y_resampled2 = smote.fit_sample(X_train, y_train)
    # pd.DataFrame(X_resampled2).to_csv('resources/data/X_resampled2.csv', index=False)
    # pd.DataFrame(y_resampled2).to_csv('resources/data/y_resampled2.csv', index=False)

    print("X_train2 ", pd.DataFrame(X_resampled2).shape)
    print("X_train2 ", pd.DataFrame(y_resampled2).shape)

    df = pd.DataFrame(y_resampled2)
    print(df.groupby('class').size())


    # sel_chi2 = SelectKBest(chi2, k=8)  # select 8 features
    # X_train_chi2 = sel_chi2.fit_transform(X_resampled2, y_resampled2)
    # print(sel_chi2.get_support())
    #
    # X_test_chi2 = sel_chi2.transform(X_test)
    # print(X_test.shape)
    # print(X_test_chi2.shape)

    # # ###############################################################################
    # # #                               4. Scale data                                 #
    # # ###############################################################################
    # sc = StandardScaler()
    # X_resampled2 = sc.fit_transform(X_resampled2)
    # X_test = sc.transform(X_test)




    # Spot Check Algorithms
    # spot_check_algorithms(X_train_chi2, y_resampled2)




    # Make predictions on validation dataset using the selected model
    thoracic_model = DecisionTreeClassifier() # MLP- 0.88, ExtraTreeClassifier-0.73,0.97,0.94,0.91,0.94, 0.94, 0.86   RF- 0.88, 0.88  GB- 0.89, 0.89  LR()- 0.88, 0.88  LogisticRegression(solver='liblinear', multi_class='ovr') - 0.92, kNN- 0.87, 0.92, 0.84   DT- 0.94, 0.94, 0.89, 0.94  SVC(gamma='auto') - 0.94, MultinomialNB() - 0.88
    # models2 =  VotingClassifier(
    #     estimators=[('rf', random_forest), ('knn', KNeighborsClassifier(n_neighbors=5)), ('NB', GaussianNB())],
    #     voting='hard') # 0.74


    # Train the final model
    thoracic_model = thoracic_model.fit(X_resampled2, y_resampled2)

    # Evaluate the final model on the training set
    predictions = thoracic_model.predict(X_resampled2)
    print_evaluation_results(y_resampled2, predictions)

    # Evaluate the final model on the test set
    predictions = thoracic_model.predict(X_test)
    print_evaluation_results(y_test, predictions, train=False)

    joblib.dump(thoracic_model, filename='../resources/models/sub_classifier_2.pkl')

