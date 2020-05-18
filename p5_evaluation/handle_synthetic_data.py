from IPython.core.pylabtools import figsize
from dython.nominal import associations
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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from p1_eda.eda import get_details, visualize_class_distribution, visualise_feature_distribution, check_duplicates
from p2_preprocessing.class_imbalance import balance_classes
from p2_preprocessing.data_cleansing import impute_missing_values, perform_one_hot_encoding
from p2_preprocessing.feature_selection import select_features
from p3_classification.baseline import get_baseline_performance
# from p3_classification.spot_check import spot_check_algorithms
from p5_evaluation.model_evaluation import evaluate_models, get_scores

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)





# Replace String labels with integres
def encode_string_labels():
    data_frame['class'].replace('\'head and neck\'', 'head and neck', inplace=True)
    data_frame['class'].replace('\'corpus uteri\'', 'corpus uteri', inplace=True)
    data_frame['class'].replace('\'cervix uteri\'', 'cervix uteri', inplace=True)
    data_frame['class'].replace('\'duoden and sm.int\'', 'duoden and sm.int', inplace=True)
    data_frame['class'].replace('\'salivary glands\'', 'salivary glands', inplace=True)
    print("After replacing\n", data_frame.head(10))

    age_mapping = {'<30': 1, '30-59': 2, '>=60': 3}
    sex_mapping = {'male': 1, 'female': 2}
    histo_mapping = {'epidermoid': 1, 'adeno': 2, 'anaplastic': 3}
    dof_mapping = {'well': 1, 'fairly': 2, 'poorly': 3}

    data_frame['age'] = data_frame['age'].map(age_mapping)
    data_frame['sex'] = data_frame['sex'].map(sex_mapping)
    data_frame['histologic-type'] = data_frame['histologic-type'].map(histo_mapping)
    data_frame['degree-of-diffe'] = data_frame['degree-of-diffe'].map(dof_mapping)

    boolean_mapping = {'yes': 1, 'no': 2}
    # columns = [ 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver', 'brain', 'skin', 'neck', 'supraclavicular',
    #          'axillar', 'mediastinum', 'abdominal' ]
    data_frame['bone'] = data_frame['bone'].map(boolean_mapping)
    data_frame['bone-marrow'] = data_frame['bone-marrow'].map(boolean_mapping)
    data_frame['lung'] = data_frame['lung'].map(boolean_mapping)
    data_frame['pleura'] = data_frame['pleura'].map(boolean_mapping)
    data_frame['peritoneum'] = data_frame['peritoneum'].map(boolean_mapping)
    data_frame['liver'] = data_frame['liver'].map(boolean_mapping)
    data_frame['brain'] = data_frame['brain'].map(boolean_mapping)
    data_frame['skin'] = data_frame['skin'].map(boolean_mapping)
    data_frame['neck'] = data_frame['neck'].map(boolean_mapping)
    data_frame['supraclavicular'] = data_frame['supraclavicular'].map(boolean_mapping)
    data_frame['axillar'] = data_frame['axillar'].map(boolean_mapping)
    data_frame['mediastinum'] = data_frame['mediastinum'].map(boolean_mapping)
    data_frame['abdominal'] = data_frame['abdominal'].map(boolean_mapping)

    class_mapping = {'lung': 1, 'head and neck': 2, 'esophagus': 3, 'thyroid': 4, 'stomach': 5, 'duoden and sm.int': 6,
                     'colon': 7, 'rectum': 8, 'anus': 9, 'salivary glands': 10, 'pancreas': 11, 'gallbladder': 12,
                     'liver': 13,
                     'kidney': 14, 'bladder': 15, 'testis': 16, 'prostate': 17, 'ovary': 18, 'corpus uteri': 19,
                     'cervix uteri': 20, 'vagina': 21, 'breast': 22}

    for key in (class_mapping):
        data_frame['class'].replace(key, class_mapping[key], inplace=True)

    print(data_frame.head(10))










######################################################  Configurations  ################################################
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

DATA_DIRECTORY_PATH = "../resources/datasets/"
DATASET_FILE_NAME = "BNG_primary_tumor.csv"
DATASET_PATH = DATA_DIRECTORY_PATH + DATASET_FILE_NAME
TARGET_NAME = "class"



######################################################  Import Data  ###################################################
data_frame =  pd.read_csv(DATASET_PATH, na_values='?')



#########################################################  EDA  ########################################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  EDA  !!!!!!!!!!!!!!!!!!!!!!!!\n")
get_details(data_frame)
# visualize_class_distribution(data_frame, "class")
# visualise_feature_distribution(data_frame)
is_duplicated = check_duplicates(data_frame)



###################################################  Data Preprocessing  ###############################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  DATA PREPROCESSING  !!!!!!!!!!!!!!!!!!!!!!!!\n")

# Encode string labels
encode_string_labels()

# visualize_class_distribution(data_frame, "class")
# visualise_feature_distribution(data_frame)


# Drop duplicate records if exist
if is_duplicated:
    data_frame.drop_duplicates(inplace=True)
    print("Dropped duplicate records. Size after dropping duplicates: ", data_frame.shape)


# Returns a DataFrame of the correlation/strength-of-association between all features
# associations(data_frame, theil_u=True)


# One Hot Encoding
# columns_to_encode = ['sex', 'histologic-type', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver', 'brain',
#                        'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal']
# data_frame = perform_one_hot_encoding(data_frame, columns_to_encode)


data_frame = data_frame.loc[data_frame['class'] != 9].sample(frac=1,random_state=42)
normalized_df = pd.concat([data_frame])

# Write pre-processed data to a file
data_frame.to_csv(DATA_DIRECTORY_PATH + 'pre_processed_bng_data.csv', index=False)






# load data
data_frame =  pd.read_csv(DATA_DIRECTORY_PATH + 'pre_processed_bng_data.csv', na_values='?')

# Shuffle the Dataset.
shuffled_df = data_frame.sample(frac=1,random_state=4)

# Randomly select 6 observations from the testis (minority class)
SAMPLE_SIZE = 80
# duo_sm_df =  shuffled_df.loc[shuffled_df['class'] == 6].sample(n=SAMPLE_SIZE,random_state=42)
# testis_df =  shuffled_df.loc[shuffled_df['class'] == 16].sample(n=SAMPLE_SIZE,random_state=42)
# vagina_df =  shuffled_df.loc[shuffled_df['class'] == 21].sample(n=SAMPLE_SIZE,random_state=42)
# sal_gla_df =  shuffled_df.loc[shuffled_df['class'] == 10].sample(n=SAMPLE_SIZE,random_state=42)
# uri_b_df =  shuffled_df.loc[shuffled_df['class'] == 15].sample(n=SAMPLE_SIZE,random_state=42)
# cerv_uter_df =  shuffled_df.loc[shuffled_df['class'] == 20].sample(n=SAMPLE_SIZE,random_state=42)
#
# # Concatenate both dataframes again
# synthetic_df = pd.concat([ duo_sm_df, testis_df, vagina_df, sal_gla_df, uri_b_df, cerv_uter_df ])
# print(synthetic_df.head())

synthetic_df = None
classes_to_balance = [2,  3,  4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
for i in range(len(classes_to_balance)):
    synthetic_instances_df = shuffled_df.loc[shuffled_df['class'] == classes_to_balance[i]].sample(n=SAMPLE_SIZE,random_state=42)
    synthetic_df = pd.concat([ synthetic_df, synthetic_instances_df ])




# Concatenate both dataframes again
# synthetic_df = pd.concat([ duo_sm_df, testis_df, vagina_df, sal_gla_df, uri_b_df, cerv_uter_df ])
print(synthetic_df.head())
print(synthetic_df.groupby('class').size())

# Write all the synthetic samples to a file
synthetic_df.to_csv(DATA_DIRECTORY_PATH + 'synthetic_minority_samples.csv', index=False)


#plot the dataset after the undersampling
plt.figure(figsize=(8, 8))
sns.countplot('class', data=normalized_df)
plt.title('Balanced Classes')
plt.show()