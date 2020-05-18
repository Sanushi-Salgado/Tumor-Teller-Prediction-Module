# Author: Sanushi Salgado

import warnings
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
# from evaluation.model_evaluation import print_evaluation_results
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from yellowbrick.classifier import confusion_matrix

from p1_eda.eda import get_details, check_duplicates, perform_correspondence_analysis
from p2_preprocessing.data_cleansing import impute_missing_values, perform_one_hot_encoding
from p2_preprocessing.feature_selection import get_feature_correlations
from p3_classification.extra_peritoneum_classifier import extra_peritoneum_region_classifier
from p3_classification.intra_peritoneum_classifier import intra_peritoneum_region_classifier
from p3_classification.sk_hc import classify_digits
from p3_classification.spot_check import spot_check_algorithms
from p3_classification.thoracic_classifier import thoracic_region_classifier
from p3_classification.upper_region_classifier import upper_region_classifier
from p5_evaluation.model_evaluation import print_evaluation_results, plot_confusion_matrix

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)




################################################# Configurations #######################################################
DATA_DIRECTORY_PATH = "../resources/datasets/"
FINAL_MODELS_PATH = "../resources/models/"
DATASET_FILE_NAME = "primary-tumor-with-region.csv"
UPPER_REGION_DATASET = "upper_region.csv"
THORACIC_REGION_DATASET = "thoracic_region.csv"
INTRA_PERITONEUM_REGION_DATASET = "intra_peritoneum_region.csv"
EXTRA_PERITONEUM_REGION_DATASET = "extra_peritoneum_region.csv"

UR_CLASSES = ['2', '4', '10']
TR_CLASSES = ['1', '22']
IP_CLASSES = ['3', '5', '7', '11', '12', '13', '6']
# IP_CLASSES = ['3', '5', '7', '11', '12', '13']
EP_CLASSES = ['8', '14', '17', '18', '19', '15', '16', '20', '21']
# EP_CLASSES = ['8', '14', '17', '18', '19', '15',  '20']

TOP_LEVEL_TARGET = 'region'
SECOND_LEVEL_TARGET = 'class'

RANDOM_STATE = 42






# Creates datasets for the second level sub classifiers
def create_datasets(dataframe):
    # # Remove all classes with only 1 instance
    # class_filter = data_frame.groupby(SECOND_LEVEL_TARGET)
    # data_frame = class_filter.filter(lambda x: len(x) > 1)
    # print("2nd Level - Class Count", data_frame.groupby(SECOND_LEVEL_TARGET).size())


    # Filter out the instances according to their regions
    # To  ensure that each sub classifier is trained and tested only on its child classes
    upper_region_dataframe = data_frame[data_frame[SECOND_LEVEL_TARGET].isin(UR_CLASSES)]
    thoracic_region_dataframe = data_frame[data_frame[SECOND_LEVEL_TARGET].isin(TR_CLASSES)]
    intra_peritoneum_region_dataframe = data_frame[data_frame[SECOND_LEVEL_TARGET].isin(IP_CLASSES)]
    extra_peritoneum_region_dataframe = data_frame[data_frame[SECOND_LEVEL_TARGET].isin(EP_CLASSES)]

    # Write datasets to separate file
    upper_region_dataframe.to_csv(DATA_DIRECTORY_PATH + 'upper_region.csv', index=False)
    thoracic_region_dataframe.to_csv(DATA_DIRECTORY_PATH + 'thoracic_region.csv', index=False)
    intra_peritoneum_region_dataframe.to_csv(DATA_DIRECTORY_PATH + 'intra_peritoneum_region.csv', index=False)
    extra_peritoneum_region_dataframe.to_csv(DATA_DIRECTORY_PATH + 'extra_peritoneum_region.csv', index=False)






import matplotlib.pyplot as plt
def plot(x, y , predicted):
    plt.scatter(x, y, color='black')
    plt.plot(x, predicted,'-r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



# https://gist.github.com/naught101/4d1661b773471895fc52
def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError(
            "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables]) -
            entropy(all_vars, k=k))





def classify_by_region(data_frame):
    get_details(data_frame)
    print("Before Oversampling By Region\n", data_frame.groupby('region').size())
    # sns.countplot(data_frame['region'], label="Count")
    # plt.show()

    # sns.heatmap(data_frame.drop('region', axis=1), cmap='cool', annot=True)
    # plt.show()

    # get_feature_correlations(data_frame, plot=True, return_resulst=False)


    X = data_frame.drop(['region', 'class'], axis=1)  # Features - drop class from features - 'age', 'sex',
    y = data_frame['region']  # Labels


    mutual_info = mutual_info_classif(X, y, discrete_features='auto')
    print("mutual_info: ", mutual_info)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    # X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, shuffle=True)


    sm = BorderlineSMOTE()
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    print("After Oversampling By Region\n", (pd.DataFrame(y_resampled)).groupby('region').size())
    # X_resampled.to_csv('resources/data/X_resampled.csv', index=False)
    # y_resampled.to_csv('resources/data/y_resampled.csv', header=['region'], index=False)



    ###############################################################################
    #                               4. Scale data                                 #
    ###############################################################################
    # sc = StandardScaler()
    # X_resampled = sc.fit_transform(X_resampled)
    # X_test = sc.transform(X_test)



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
    # # plt.show()
    #
    sel_chi2 = SelectKBest(chi2, k='all')  # chi 10 - 0.64, 0.63, 0.60
    X_train_chi2 = sel_chi2.fit_transform(X_resampled, y_resampled)
    X_test_chi2 = sel_chi2.transform(X_test)



    # Spot Check Algorithms
    # spot_check_algorithms(X_resampled, y_resampled)



    # models = [SVC(kernel='poly'), RandomForestClassifier(),  GradientBoostingClassifier()]
    # for i in range(len(models)):
    #     # Get the final model
    #     parent_model = models[i] # LR(multiclass-ovr) -0.66, 0.67, 0.67, 0.69, 0.69, 0.68  MLP wid fs - 0.65, 0.69, 0.70,   GB - 0.67, without fs 0.62, 0.61,    DT - 0.58,   RF - 0.67,  multi_LR - wid fs 0.64 , voting - 0.60
    #
    #     # Train the final model
    #     parent_model.fit(X_resampled, y_resampled)
    #
    #     # Evaluate the final model on the training set
    #     predictions = parent_model.predict(X_resampled)
    #     print_evaluation_results(y_resampled, predictions)
    #
    #     # Evaluate the final model on the test set
    #     predictions = parent_model.predict(X_test)
    #     print_evaluation_results(y_test, predictions, train=False)



    # mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes = [100]*5, random_state=42))

    pipeline = Pipeline(
            [
                # ('selector', SelectKBest(f_classif)),
                ('model',  RandomForestClassifier(n_jobs = -1) )
            ]
    )

    # Perform grid search on the classifier using f1 score as the scoring method
    grid_obj = GridSearchCV(
            estimator= GradientBoostingClassifier(),
            param_grid={
                # 'selector__k': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                'n_estimators': [10, 20, 30],
                'max_depth': [6, 10, 20, 30],
                # 'max_depth': [1, 10, 20, 30],
                'min_samples_split': [1, 10, 100]
                # 'model__n_estimators': np.arange(10, 200, 10)
                # 'C': [1, 10, 100]
            },

            n_jobs=-1,
            scoring="f1_micro",
            cv=5,
            verbose=3
    )

    # Fit the grid search object to the training data and find the optimal parameters
    grid_fit =  grid_obj.fit(X_resampled, y_resampled)

    # Get the best estimator
    best_clf = grid_fit.best_estimator_
    print(best_clf)




    # Get the final model
    parent_model = best_clf # LR(multiclass-ovr) -0.66, 0.67, 0.67, 0.69, 0.69, 0.68  MLP wid fs - 0.65, 0.69, 0.70,   GB - 0.67, without fs 0.62, 0.61,    DT - 0.58,   RF - 0.67,  multi_LR - wid fs 0.64 , voting - 0.60

    t0 = time()
    # Train the final model
    parent_model.fit(X_resampled, y_resampled)
    print("training time:", round(time() - t0, 3), "s")

    # Evaluate the final model on the training set
    train_predictions = parent_model.predict(X_resampled)
    print_evaluation_results(y_resampled, train_predictions)

    t0 = time()
    # Evaluate the final model on the test set
    test_predictions = parent_model.predict(X_test)
    print("predicting time:", round(time() - t0, 3), "s")

    print_evaluation_results(y_test, test_predictions, train=False)
    confusion_matrix(parent_model, X_resampled, y_resampled, X_test, y_test)

    # Plot normalized confusion matrix
    # fig = plt.figure()
    # fig.set_size_inches(8, 8, forward=True)
    # # fig.align_labels()
    # plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4"], normalize=False, title='Normalized confusion matrix')


    # probs = parent_model.predict_proba(X_test)
    # print("Prediction probabilities for Region\n", probs)
    # plotConfusionMatrix(X_test, y_test, ['1', '2', '3', '4'])

    # joblib.dump(parent_model, filename='../resources/models/parent_classifier.pkl')










#########################################################  Load Dataset  ###############################################
data_frame = pd.read_csv(DATA_DIRECTORY_PATH + DATASET_FILE_NAME, na_values='?', dtype='category')



#########################################################  EDA  ########################################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  EDA  !!!!!!!!!!!!!!!!!!!!!!!!\n")
get_details(data_frame)
print("Class count\n", data_frame.groupby(SECOND_LEVEL_TARGET).size())

# visualize_class_distribution(data_frame, "class")
# visualise_feature_distribution(data_frame)

# Check if duplicate records exist
is_duplicated = check_duplicates(data_frame)



###################################################  Data Preprocessing  ###############################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  DATA PREPROCESSING  !!!!!!!!!!!!!!!!!!!!!!!!\n")

# Impute missing values
data_frame = impute_missing_values(data_frame, "most_frequent")
print(data_frame.head(20))
print(data_frame.isnull().sum().sum())


# Get the correlation matrix
# get_feature_correlations(data_frame, plot=True, return_resulst=True)


# # Drop duplicate records if exist
# if is_duplicated:
#     data_frame.drop_duplicates(inplace=True)
#     print("Dropped duplicate records. Size after dropping duplicates: ", data_frame.shape)


# CA
# perform_correspondence_analysis(data_frame)


# One Hot Encoding
columns_to_encode = ['sex', 'histologic-type', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver',
                     'brain', 'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal',
                     'small-intestine']
data_frame = perform_one_hot_encoding(data_frame, columns_to_encode)

# Pre-prcoessed dataset
pre_processed_data = data_frame




##################################### Create separate datasets ##################################
create_datasets(pre_processed_data)


##################################### Parent Classifier ##################################
# Classifies items between upper region, thoracic region, intra peritoneum & extra peritoneum region
classify_by_region(pre_processed_data)


##################################### Sub Classifier 01 ##################################
# Classifies between upper region organs i.e thyroid, salivary glands and head & neck
upper_region_classifier()


# ##################################### Sub Classifier 02 ##################################
# Classifies between thoracic region organs i.e thyroid, salivary glands and head & neck
thoracic_region_classifier()


# ##################################### Sub Classifier 03 ##################################
# Classifies between intra_peritoneum organs i.e
# intra_peritoneum_region_classifier()


# ##################################### Sub Classifier 04 ##################################
# Classifies between extra peritoneum organs i.e
extra_peritoneum_region_classifier()








