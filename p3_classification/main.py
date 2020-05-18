from dython.nominal import associations
from imblearn.ensemble import BalancedBaggingClassifier
from pandas.plotting import scatter_matrix
from sklearn import preprocessing

import pandas as pd
import os.path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from p1_eda.eda import get_details, visualize_class_distribution, visualise_feature_distribution, check_duplicates
from p2_preprocessing.class_imbalance import balance_classes
from p2_preprocessing.data_cleansing import impute_missing_values, perform_one_hot_encoding
from p2_preprocessing.feature_selection import select_features, get_feature_correlations
from p3_classification.baseline import get_baseline_performance
# from p3_classification.spot_check import spot_check_algorithms
from p5_evaluation.model_evaluation import evaluate_models, get_scores

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)



######################################################  Configurations  ################################################
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

DATA_DIRECTORY_PATH = "../resources/datasets/"
DATASET_FILE_NAME = "primary_tumor.csv"
DATASET_PATH = DATA_DIRECTORY_PATH + DATASET_FILE_NAME
TARGET_NAME = "class"



######################################################  Import Data  ###################################################
data_frame =  pd.read_csv(DATASET_PATH, na_values='?', dtype='category')



#########################################################  EDA  ########################################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  EDA  !!!!!!!!!!!!!!!!!!!!!!!!\n")
get_details(data_frame)
# visualize_class_distribution(data_frame, "class")
# visualise_feature_distribution(data_frame)
is_duplicated = check_duplicates(data_frame)



###################################################  Data Preprocessing  ###############################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  DATA PREPROCESSING  !!!!!!!!!!!!!!!!!!!!!!!!\n")

# Impute missing values
data_frame = impute_missing_values(data_frame, "most_frequent")

# import seaborn as sns
# sns.pairplot(data_frame,kind='scatter')
# plt.show()


# Drop duplicate records if exist
if is_duplicated:
    data_frame.drop_duplicates(inplace=True)
    print("Dropped duplicate records. Size after dropping duplicates: ", data_frame.shape)



# Imbalanced DataFrame Correlation
# Sample figsize in inches
# fig, ax = plt.subplots(figsize=(20,10))
# # Returns a DataFrame of the correlation/strength-of-association between all features
# corr = associations(data_frame, theil_u=True, plot=False, return_results=True)
# sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
# ax.set_title("Imbalanced Correlation Matrix", fontsize=14)
# plt.show()




# One Hot Encoding
columns_to_encode = ['sex', 'histologic-type', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver', 'brain',
                       'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal']
data_frame = perform_one_hot_encoding(data_frame, columns_to_encode)


# Write pre-processed data to a file
data_frame.to_csv('../resources/datasets/pre_processed_data.csv', index=False)




# Separating input features & target
input_features = data_frame.drop(TARGET_NAME, axis=1)  # Features
target = data_frame[TARGET_NAME]  # Labels
print("Features: ", input_features.shape)
print("Labels: ", target.shape)





#################################################  Train Test Split  ###################################################
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows -benchmark results, shuffle
X_train, X_test, y_train, y_test = train_test_split(input_features, target, test_size=0.3, random_state=42, shuffle=True)
# X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
# stratify=target - ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any
# class cannot be less than 2. -  train-test split with stratification -Here you are dividing input data X, and targets y
#  into training set(Xtrain,ytrain) and test set(Xtest and ytest) while preserving distribution in the input data.
# https://www.kaggle.com/c/avazu-ctr-prediction/discussion/11374

data_partitions = { "training_features" : X_train,
                    "testing_features": X_test,
                    "training_labels":  y_train,
                    "testing_labels": y_test
                  }

for key in data_partitions:
    file_exists = os.path.isfile( DATA_DIRECTORY_PATH + '%s%s' % (key, '.csv') )
    if not file_exists:
        file = open( (DATA_DIRECTORY_PATH + '%s%s' % (key, '.csv')), mode='w+' )
    pd.DataFrame( data_partitions[key] ).to_csv(( DATA_DIRECTORY_PATH + '%s%s' % (key, '.csv') ), index=False)
    # Display sizes of datasets partitions
    print('%s %s' % (key, data_partitions[key].shape) )





########################################  Establish Baseline in Performance  ###########################################
print('\n\n!!!!!!!!!!!!!!!!!!!!!!!   Baseline in Performance  !!!!!!!!!!!!!!!!!!!!!!!\n')
# get_baseline_performance(X_train, y_train, X_test, y_test)



##############################################  Spot Checking Algorithms  ##############################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  ALGORITHM SELECTION  !!!!!!!!!!!!!!!!!!!!!!!!\n")
# spot_check_algorithms(X_train, y_train)



# # #Create an object of the classifier.
# bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(),
#                                 sampling_strategy='auto',
#                                 replacement=False,
#                                 random_state=0)
#
# #Train the classifier.
# bbc.fit(X_train, y_train)
# preds = bbc.predict(X_train)
# print("Balanced Bagging Classifier")
# print("Training Performance\n", f1_score(y_train, preds, average='micro'))
# preds = bbc.predict(X_test)
# print("Testing Performance\n", f1_score(y_test, preds, average='micro'))



##################################################  Class Biasness  ####################################################
X_resampled, y_resampled = balance_classes(X_train, y_train)

# Balanced DataFrame Correlation
# Sample figsize in inches
# fig, ax = plt.subplots(figsize=(20,10))
# data_frame = pd.concat([X_train, X_test, y_train, y_test])
# # Returns a DataFrame of the correlation/strength-of-association between all features
# corr = associations(data_frame, theil_u=True, plot=False, return_results=True)
# sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
# ax.set_title("After resampling: Correlation Matrix", fontsize=14)
# plt.show()



####################################################  FEATURE SELECTION  ###############################################
# methods = [chi2, mutual_info_classif]
# X_train_fs = None
# X_test_fs = None
# for i in range(len(methods)):
#     X_train_fs, X_test_fs, fs = select_features(X_resampled, y_resampled, X_test, methods[i])
#     # what are scores for the features
#     for i in range(len(fs.scores_)):
#         print('Feature %d: %f' % (i, fs.scores_[i]))
#     # plot the scores
#     plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
#     plt.show()





####################################################  MODEL EVALUATION  ################################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  MODEL EVALUATION  !!!!!!!!!!!!!!!!!!!!!!!!\n")

# Now we will compare five different machine learning models
models_to_evaluate = { 'Decision Tree' : DecisionTreeClassifier(),
                       'Random Forest': RandomForestClassifier(),
                       'Gradient Boosting': GradientBoostingClassifier(),
                       'Extra Trees': ExtraTreesClassifier() }


evaluate_models(models_to_evaluate, X_resampled, y_resampled, X_test, y_test, "F-Measure", "Classification Models", "")
# visualize_results("F-Measure", "Classification Models", "")



