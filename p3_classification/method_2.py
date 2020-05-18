from time import time
from typing import Counter

from dython.model_utils import roc_graph
from sklearn import metrics
import logging

import dython as dython
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as graph
from IPython.core.pylabtools import figsize
from itertools import cycle

# replace 2 value of each specified column with 0
import ss as ss
from Tools.demo.sortvisu import distinct
from dython.nominal import cramers_v, associations
import imblearn.metrics
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.under_sampling import ClusterCentroids, TomekLinks
from jedi.refactoring import inline
from matplotlib import ticker
from pandas.core.algorithms import duplicated
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, \
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, SGDClassifier, \
    PassiveAggressiveClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_similarity_score, confusion_matrix, \
    classification_report, mean_absolute_error, SCORERS, f1_score, cohen_kappa_score, precision_score, recall_score, \
    roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold, \
    GridSearchCV, KFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# latex parameter
font = {
    'family': 'serif',
    'serif': ['Computer Modern Roman'],
    'weight' : 'regular',
    'size'   : 14
    }

plt.rc('font', **font)




def make_boolean(data_frame):
    cols_to_replace = ['binaryClass']
    for col in cols_to_replace:
        data_frame[col].replace('1', '1', inplace=True)
        data_frame[col].replace('2', '0', inplace=True)



def impute_missing_values(data_frame):
    columns = data_frame.columns
    imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent", fill_value=None, verbose=0, copy=True)
    imputed = imp.fit_transform(data_frame)
    data_frame = pd.DataFrame(imputed, columns=columns)
    print("After imputing\n", data_frame.isnull().sum())



def handle_duplicate_records(data_frame):
    print("############## Duplicate records ############## ")
    duplicate_records = data_frame[data_frame.duplicated()]
    print("Before dropping duplicates ", duplicate_records.shape)
    print(duplicate_records.head(35))
    data_frame.drop_duplicates(inplace=True)
    print("After dropping duplicates ", data_frame.shape)
    return data_frame




def visualize_class_distribution(data_frame):
    sns.countplot(data_frame['class'], label="No of instances") # - method 1

    # plt.figure(figsize=(6,6))
    # Y = data_frame["binaryClass"]
    # total = len(Y)*1
    # majority_count = len(data_frame[data_frame['binaryClass'] == '1'])
    # ax=sns.countplot(x="binaryClass", data=data_frame)
    #
    # for p in ax.patches:
    #     ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))
    #
    # #put 11 ticks (therefore 10 steps), from 0 to the total nusmber of rows in the dataframe
    # # ax.yaxis.set_ticks(np.linspace(0, total, 11)) # gives 309 at d top of d y axis
    # ax.yaxis.set_ticks(np.linspace(0, majority_count, 11))
    # #adjust the ticklabel to the desired format, without changing the position of the ticks.
    # # ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total)) - y axis values r display as %s
    # ax.set_yticklabels(map('{:.0f}'.format, ax.yaxis.get_majorticklocs()))
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=360, ha="right")
    # # Use a LinearLocator to ensure the correct number of ticks
    # # And use a MultipleLocator to ensure a tick spacing of 10
    # # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    # # ax.legend(labels=["lung","salivary g", "Pancreas", "gall bladder", "liver"])
    # plt.title('Distribution of Classes')
    # plt.ylabel("No of instances")
    plt.show()





# Visualizes the feature distribution with box plots.
# source - https://towardsdatascience.com/designing-a-feature-selection-pipeline-in-python-859fec4d1b12
def visualise_feature_distribution(data):
    # Set graph style
    sns.set(font_scale=0.75)
    sns.set_style({'axes.facecolor': '1.0', 'axes.edgecolor': '0.85', 'grid.color': '0.85',
                   'grid.linestyle': '-', 'axes.labelcolor': '0.4', 'xtick.color': '0.4',
                   'ytick.color': '0.4', 'axes.grid': False})

    # Create box plots based on feature type
    # Set the figure size
    f, ax = plt.subplots(figsize=(9, 14))
    sns.boxplot(data=data, orient="h", palette="Set2")  # X
    # Set axis label
    plt.xlabel('Feature Value')
    # Tight layout
    f.tight_layout()
    # Save figure
    f.savefig('Box Plots.png', dpi=1080)
    plt.show()






def visualize_age():
    # sns.countplot(data_frame['bone'], label="Count")
    # sns.catplot(x="bone", y="class",  kind="swarm", data=data_frame)
    # plt.show()

    sns.scatterplot(x="age", y="class",  data=data_frame,  marker="x", label="data")
    plt.legend(labels=["1 - <30", "2 - 30-59", "3 - >=60"])
    plt.show()

    sns.catplot(x="age", y="class", kind="swarm", data=data_frame)
    plt.legend(labels=["1 - <30", "2 - 30-59", "3 - >=60"])
    plt.show()

    sns.catplot(x="brain", y="class", kind="swarm", data=data_frame)
    plt.legend(labels=["0 - no", "1 - yes", ])
    plt.show()

    sns.catplot(x="skin", y="class", kind="swarm", data=data_frame)
    plt.legend(labels=["0 - no", "1 - yes", ])
    plt.show()

    sns.catplot(x="class", y="neck", kind="swarm", data=data_frame)
    plt.legend(labels=["0 - no", "1 - yes", ])
    plt.show()






def perform_one_hot_encoding(data_frame):
    cols_to_replace = ['sex', 'histologic-type', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver', 'brain',
                       'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal']
    data_frame = pd.get_dummies( data=data_frame, columns=cols_to_replace, drop_first=True )
    print("After OHE\n", data_frame.head())
    print(data_frame.info())





def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()





def check_linear_separability():
    plt.figure(figsize=(8, 8))
    x = data_frame.drop(['class'], axis=1)  # Features
    y = data_frame['class'] == '2'  # Labelss
    perceptron = Perceptron(random_state=0)
    perceptron.fit(x, y)
    predicted = perceptron.predict(x)

    cm = confusion_matrix(y, predicted)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative', 'Positive']
    plt.title('Perceptron Confusion Matrix - Entire Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.show()





def calculate_baseline():
    # evaluate naive
    naive = DummyClassifier(strategy='most_frequent')
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # n_scores = cross_val_score(naive, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # print('Baseline: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    # # evaluate model
    # model = RidgeClassifier(alpha=0.2)
    # steps = [('s', StandardScaler()), ('m', model)]
    # pipeline = Pipeline(steps=steps)
    # m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # print('Good: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))





# def calculate_null_accuracy():
#     null_accuracy = y_train.value_counts().head(1) / len(y_train)
#     print("Null Accuracy ", null_accuracy)
#
#     null_accuracy = y_test.value_counts().head(1) / len(y_test)
#     print("Null Accuracy ", null_accuracy)





# evaluate a model
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
    return scores





# define models to test
def get_models():
    models, names = list(), list()
    # Uniformly Random Guess
    models.append(DummyClassifier(strategy='uniform'))
    names.append('Uniform')
    # Prior Random Guess
    models.append(DummyClassifier(strategy='stratified'))
    names.append('Stratified')
    # Majority Class: Predict 0
    models.append(DummyClassifier(strategy='most_frequent'))
    names.append('Majority')
    # Minority Class: Predict 1
    models.append(DummyClassifier(strategy='constant', constant=1))
    names.append('Minority')
    # Class Prior
    models.append(DummyClassifier(strategy='prior'))
    names.append('Prior')
    return models, names







#Function to print best hyperparamaters:
def print_best_params(gd_model):
    param_dict = gd_model.best_estimator_.get_params()
    model_str = str(gd_model.estimator).split('(')[0]
    print("\n*** {} Best Parameters ***".format(model_str))
    for k in param_dict:
        print("{}: {}".format(k, param_dict[k]))
    print()






def display_results(actual_values, predictions):
        print( "Accuracy Score ", accuracy_score(actual_values, predictions) )
        print( "Train Error ", 1 - accuracy_score(actual_values, predictions) )

        print( 'Precision macro is:' + str(round(precision_score(actual_values, predictions, average='macro'), 2)))
        print( 'Precision micro is:' + str(round(precision_score(actual_values, predictions, average='micro'), 2)))

        print( 'Recall macro is:' + str(round(recall_score(actual_values, predictions, average='macro'), 2)))
        print( 'Recall micro is:' + str(round(recall_score(actual_values, predictions, average='micro'), 2)))

        print( 'F1 macro:', str(round(f1_score(actual_values, predictions, average='macro'), 2)))
        print( 'F1 weighted:', str(round(f1_score(actual_values, predictions, average='weighted'), 2)))
        model_f1 = round(f1_score(actual_values, predictions, average='micro'), 2)
        print( 'F1 micro:', str(round(f1_score(actual_values, predictions, average='micro'), 2)))

        print( 'Kappa', cohen_kappa_score(actual_values, predictions))

        print( "Hamming Loss ", hamming_loss(actual_values, predictions) )
        print( "Jaccard Similarity Score ", jaccard_similarity_score(actual_values, predictions) )

        # print( 'ROC area', metrics.roc_curve(y, predictions, pos_label="1") )

        # cm = confusion_matrix(actual_values, predictions)
        # print( "Confusion Matrix: \n", cm)
        # plt.subplots(figsize=(12, 9))
        # sns.heatmap(cm, annot=True)
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.show()

        print(classification_report(actual_values, predictions))
        print(classification_report_imbalanced(actual_values, predictions))
        print()

        # fpr, tpr, thres = roc_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label='1')
        # graph.figure(figsize=(4, 4))
        # graph.plot(fpr, tpr, label='Test')
        # graph.xlabel('FPR')
        # graph.ylabel('TPR')
        # graph.show()






def select_features(X_train, y_train):
    no_of_features = range(18)
    for i in range(no_of_features):
        # We will apply the feature selection based on X_train and y_train.
        sel_chi2 = SelectKBest(chi2, k=i)  # select 4 features
        X_train_chi2 = sel_chi2.fit_transform(X_train, y_train)
        print(sel_chi2.get_support())





def judge_model(model, name, x_test, plot=False):
        print(name)
        print('-' * 20)
        # Get baseline in performance, on the test set
        predictions = model.predict(x_test)
        print('-> F1 score:', f1_score(y_test, predictions, average='micro'))







def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()









def make_datasets(dataframe):
    df_1_instances = []
    df_2_instances = []

    column = dataframe['class']
    length = len(column)

    for row in range(length):
        value = column[row]
        # if ( value == '1' or value == '6' or value == '10' or value == '15' or value == '16' or value == '20' or value == '21' ):
        #     record = dataframe.values[row]
        #     df_1_instances.append(record)
        # elif( value != '1' and value != '6' and value != '10' and value != '15' and value != '16' and value != '20' and
        #       value != '21' ):
        #     record = dataframe.values[row]
        #     df_2_instances.append(record)
        if ( value == '1' or value == '6' or value == '16' or value == '21'):
            record = dataframe.values[row]
            df_1_instances.append(record)
        elif value != '1' or value != '6' or value == '16' or value != '21':
            record = dataframe.values[row]
            df_2_instances.append(record)

    data_frame_1 = pd.DataFrame(df_1_instances)
    data_frame_2 = pd.DataFrame(df_2_instances)


    header = ['age', 'sex', 'histologic-type', 'degree-of-diffe', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum',
              'liver', 'brain', 'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal', 'class']

    data_frame_1.to_csv('resources/data/data_frame_1.csv', header=header, index=False)
    data_frame_2.to_csv('resources/data/data_frame_2.csv', header=header, index=False)









####################################################  Load the dataset  ################################################
data_frame = pd.read_csv("resources/data/primary-tumor-with-region.csv", na_values='?', dtype='category')
data_frame.drop(['region', 'binaryClass'], axis=1, inplace=True)

make_datasets(data_frame)

data_frame_1 = pd.read_csv("resources/data/data_frame_1.csv", na_values='?', dtype='category')
data_frame_2 = pd.read_csv("resources/data/data_frame_2.csv", na_values='?', dtype='category')

print('Min Max Dataframe')
print(data_frame_1.head(10))

print('Medium Dataframe')
print(data_frame_2.head(10))



###############################################  Exploratory Data Analysis  ###########################################
print('\n\n!!!!!!!!!!!!!!!!!!!!!!!   Exploratory Data Analysis  !!!!!!!!!!!!!!!!!!!!!!!\n')
print(data_frame_1.shape)
print(data_frame_1.head(10))
print(data_frame_1.info())
print(data_frame_1.describe())


#missing data - https://www.kaggle.com/funkegoodvibe/comprehensive-data-exploration-with-python
print("Total no of missing values ", data_frame_1.isnull().values.sum())
total = data_frame_1.isnull().sum().sort_values(ascending=False)
percent = (data_frame_1.isnull().sum()/data_frame_1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))


# Data Visualization
# visualise_feature_distribution(data_frame)
visualize_class_distribution(data_frame_1)







#################################################  Data Preprocessing  #########################################
# Drop duplicate rows
data_frame_1 = handle_duplicate_records(data_frame_1) # (30, 19)
columns = data_frame_1.columns


# visualise_feature_distribution(data_frame)


# data_frame.dropna(axis=0, inplace=True)
# Imputing missing values
imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent", fill_value=None, verbose=0, copy=True)
imputed = imp.fit_transform(data_frame_1)
data_frame_1 = pd.DataFrame(imputed, columns=columns)
print("After imputing\n", data_frame_1.isnull().sum())
print(data_frame_1.head(10))




# Find the correlation between features
# source - https://github.com/shakedzy/dython/blob/master/dython/examples.py
# associations(data_frame)
# associations(data_frame, theil_u=True)



# OHE
perform_one_hot_encoding(data_frame_1)


X = data_frame_1.drop(['class'], axis=1)  # Features
y = data_frame_1['class']  # Labels
print(X.shape)
print(y.shape)



#################################################  Train Test Split  ###################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
pd.DataFrame(X_train).to_csv('resources/data/training_features.csv', index=False)
pd.DataFrame(X_test).to_csv('resources/data/testing_features.csv', index=False)
# pd.DataFrame(X_validation).to_csv('resources/data/X_validation.csv', index=False)
pd.DataFrame(y_train).to_csv('resources/data/training_labels.csv', index=False)
pd.DataFrame(y_test).to_csv('resources/data/testing_labels.csv', index=False)
# pd.DataFrame(y_validation).to_csv('resources/data/y_validation.csv', index=False)



# Read in data into dataframes
train_features = pd.read_csv('resources/data/training_features.csv')
test_features = pd.read_csv('resources/data/testing_features.csv')
train_labels = pd.read_csv('resources/data/training_labels.csv')
test_labels = pd.read_csv('resources/data/testing_labels.csv')

# Display sizes of data
print('Training Feature Size: ', train_features.shape)
print('Testing Feature Size:  ', test_features.shape)
print('Training Labels Size:  ', train_labels.shape)
print('Testing Labels Size:   ', test_labels.shape)






#####################################  Establish Baseline in Performance  ########################################
print('\n\n!!!!!!!!!!!!!!!!!!!!!!!   Baseline in Performance  !!!!!!!!!!!!!!!!!!!!!!!\n')
# Baseline (AUC should be 0.5 because we're guessing even though the accuracies are different)
for strategy in ['stratified', 'most_frequent', 'prior', 'uniform', 'constant']:
    dummy = None
    if strategy == 'constant':
        dummy = DummyClassifier(strategy=strategy, constant='16', random_state=7)
    else:
        dummy = DummyClassifier(strategy=strategy, constant = None, random_state=7)
    dummy.fit(X_train, y_train)
    judge_model(dummy, 'Dummy {}'.format(strategy), X_test,  plot=False)






##############################################  Spot Checking Algorithms  ################################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  ALGORITHM SELECTION  !!!!!!!!!!!!!!!!!!!!!!!!\n")

estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svr', make_pipeline(StandardScaler(), MLPClassifier()))]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(multi_class='ovr'))

ensemble =  VotingClassifier(estimators=[('rf', RandomForestClassifier()), ('mlp', MLPClassifier()), (('NB', GaussianNB()))], voting='hard')
ensemble2 = VotingClassifier(estimators=[('mlp', MLPClassifier()), ('LR', LogisticRegression()), (('Ridge', RidgeClassifier()))], voting='hard')


# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []

# linear models
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('SVMl', SVC(kernel='linear')))
models.append(('sgd', SGDClassifier(max_iter=1000, tol=1e-3)))
models.append(('pa', PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)))

# Non linear models
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVMR', SVC(kernel='rbf')))
models.append(('NB', GaussianNB()))
models.append(('MNB',MultinomialNB()))
models.append(('SVM', SVC(kernel='linear')))
models.append(('SVMp', SVC(kernel='poly')))
models.append(('CART', DecisionTreeClassifier()))
models.append(('extra', ExtraTreeClassifier()))

# Ensemble models
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('AB', AdaBoostClassifier()))
models.append(('Bagg', BaggingClassifier()))
models.append(('Voting', ensemble))
models.append(('Ensemble2', ensemble2))
models.append(('Stacking', clf))

models.append(('MLP', MLPClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = ['f1_micro']
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed) # StratifiedKFold
    # cross validation score
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_micro')
    results.append(cv_results)
    names.append(name)
    # print the mean cross validation score
    # print the standard deviation of the data to see degree of variance in the results obtained by our model.
    print('%s: %f (%f)' % (name, cv_results.mean()*100, cv_results.std()*100))


# boxplot algorithm comparison
fig = plt.figure(figsize=(14,  12))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# Box plots of those algorithm’s accuracy distribution quite symmetrical, without outliers. - RF, GB










sm = RandomOverSampler() # You need to reduce k or increase the number of instances for the least represented class.
# sm = RandomOverSampler()
# tomeks = TomekLinks(sampling_strategy='majority')
# # X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X_resampled, y_resampled)

# steps = [('smote', sm), ('tomek', tomeks)]
# pipeline = Pipeline(steps=steps)

X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
# X_resampled, y_resampled = tomeks.fit_sample(X_resampled, y_resampled)
print('SMOTE {}'. format(Counter(y_resampled)))
X_resampled.to_csv('resources/data/X_resampled.csv', index=False)
y_resampled.to_csv('resources/data/y_resampled.csv', header=['binaryClass'], index=False)








####################################################  MODEL EVALUATION  ################################################
print("\n\n!!!!!!!!!!!!!!!!!!!!!!!  MODEL EVALUATION  !!!!!!!!!!!!!!!!!!!!!!!!\n")

# Now we will compare five different machine learning models
models_to_evaluate = { 'Decision Tree' : DecisionTreeClassifier(),
                       'Random Forest': RandomForestClassifier(),
                       'Gradient Boosting': GradientBoostingClassifier(),
                       'Extra Trees': ExtraTreesClassifier(),
                        'Logistic Reg': LogisticRegression(solver='liblinear')}

f1_scores_on_test_set = []
kappa_scores_on_test_set = []

# Takes in a model, trains the model, & evaluates the model on both training & test set
for key in models_to_evaluate:
    model = models_to_evaluate[key]

    start_time = time()
    # Train the model, on training data
    model = model.fit(X_resampled, y_resampled)
    end_time = time()

    print("#############  %s  #############" %  key)
    print("---------  Training Performance  --------")
    print("Time taken to train the model (Train time):", round(end_time - start_time, 3), "s")  # the time would be round to 3 decimal in seconds
    start_time = time()
    # Make predictions on training data & evaluate
    predictions = model.predict(X_resampled)
    end_time = time()
    print("Time taken to generate predictions on training set (Prediction Time):", round(end_time - start_time, 3), "s")
    display_results(y_resampled, predictions)

    print("---------  Testing Performance  --------")
    start_time = time()
    # Make predictions on test data or unseen data & evaluate
    predictions = model.predict(X_test)
    end_time = time()
    print("Time taken to generate predictions on test set (Prediction Time):", round(end_time - start_time, 3), "s")
    display_results(y_test, predictions)
    f_score = round(f1_score(y_test, predictions, average='micro'), 2)
    f1_scores_on_test_set.append(f_score)
    kappa_score = round(cohen_kappa_score(y_test, predictions), 2)
    kappa_scores_on_test_set.append( kappa_score )



# Dataframe to hold the results
model_comparison = pd.DataFrame({ 'model': [ 'Deciion Tree', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'Logistic Reg' ],
                                  'f1': f1_scores_on_test_set
                                 })

# Horizontal bar chart of test mae# Horizontal bar chart of test mae
model_comparison.sort_values('model', ascending=False).plot(x='model', y='f1', kind='bar', color='green', edgecolor='green')

# Plot formatting
plt.style.use('fivethirtyeight')
figsize(8, 12)
plt.title("F-Measure")
# plt.yticks(size=14)
plt.xlabel('Classification Models')
# plt.xticks(size=14)
# plt.title('Model Comparison on Test set using F-Measure', size=20)
plt.show()





# Dataframe to hold the results
model_comparison = pd.DataFrame({ 'model': [ 'Deciion Tree', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'Logistic Reg' ],
                                  'kappa': kappa_scores_on_test_set
                                 })

# Horizontal bar chart of test mae# Horizontal bar chart of test mae
model_comparison.sort_values('model', ascending=False).plot(x='model', y='kappa', kind='line', marker="X")

# Plot formatting
plt.style.use('fivethirtyeight')
figsize(8, 12)
plt.title("Kappa Statistics")
# plt.yticks(size=14)
plt.xlabel('Classification Models')
# plt.xticks(size=14)
plt.show()




# ###################################################  Final Model  ######################33#####################
# final_model = LogisticRegression(solver='liblinear')
# final_model.fit(X_resampled, y_resampled)
# predictions = final_model.predict(X_test)
# print(classification_report_imbalanced(y_test, predictions))

# figsize(8, 8)
# # Density plot of the final predictions and the test values
# sns.kdeplot(predictions, label = 'Predictions')
# sns.kdeplot(y_test, label = 'True Values')
# # Label the plot
# plt.xlabel('Primary Tumor Sites')
# plt.ylabel('Density')
# plt.title('Test Values and Predictions')
# plt.show()



