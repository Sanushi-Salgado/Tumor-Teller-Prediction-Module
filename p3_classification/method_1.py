import os.path
import warnings

import matplotlib.pyplot as plt
# from dython.nominal import associations
# from imblearn.ensemble import BalancedBaggingClassifier
import pandas as pd
import seaborn as sns
from IPython.core.pylabtools import figsize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from p1_eda.eda import get_details, check_duplicates
from p2_preprocessing.class_imbalance import balance_classes
from p2_preprocessing.data_cleansing import impute_missing_values, perform_one_hot_encoding
from p5_evaluation.model_evaluation import evaluate_models

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
# load the original dataset
data_frame =  pd.read_csv(DATASET_PATH, na_values='?', dtype='category')
print(data_frame.shape)

# load data synthetic dataset
data_frame2 =  pd.read_csv(DATA_DIRECTORY_PATH + 'synthetic_minority_samples.csv', na_values='?', dtype='category')
print(data_frame2.shape)

# Check for equal distributions
figsize(8, 8)
# Density plot of the final predictions and the test values
sns.kdeplot(data_frame['brain'], label = 'original')
sns.kdeplot(data_frame2['brain'], label = 'Synthetic')
# Label the plot
plt.xlabel('Primary Tumor Sites')
plt.ylabel('Density')
plt.title('Test Values and Predictions')
plt.show()


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


# Drop duplicate records if exist
if is_duplicated:
    data_frame.drop_duplicates(inplace=True)
    print("Dropped duplicate records. Size after dropping duplicates: ", data_frame.shape)



print("Combining original dataset with synthetic samples")
data_frame = pd.concat([ data_frame, data_frame2 ])
get_details(data_frame)



# One Hot Encoding
columns_to_encode = ['sex', 'histologic-type', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver', 'brain',
                       'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal']
data_frame = perform_one_hot_encoding(data_frame, columns_to_encode)

# Shuffle the Dataset
shuffled_df = data_frame.sample(frac=0.001,random_state=4)

# Write pre-processed data to a file
data_frame.to_csv(DATA_DIRECTORY_PATH + 'pre_processed_data.csv', index=False)






#################################################  Train Test Split  ###################################################
# Separating input features & target
input_features = data_frame.drop(TARGET_NAME, axis=1)  # Features
target = data_frame[TARGET_NAME]  # Labels
target = target.astype('int')
print("Features: ", input_features.shape)
print("Labels: ", target.shape)


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





##################################################  Class Biasness  ####################################################
X_resampled, y_resampled = balance_classes(X_train, y_train)



####################################################  MODEL EVALUATION  ################################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  MODEL EVALUATION  !!!!!!!!!!!!!!!!!!!!!!!!\n")

# Now we will compare five different machine learning models
models_to_evaluate = { 'Decision Tree' : DecisionTreeClassifier(),
                       'Random Forest': RandomForestClassifier(),
                       'Gradient Boosting': GradientBoostingClassifier(),
                       'Extra Trees': ExtraTreesClassifier() }


evaluate_models(models_to_evaluate, X_resampled, y_resampled, X_test, y_test, "F-Measure", "Classification Models", "")
# visualize_results("F-Measure", "Classification Models", "")



