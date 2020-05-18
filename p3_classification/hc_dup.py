# Author: Sanushi Salgado

import warnings
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
# from evaluation.model_evaluation import print_evaluation_results
from boruta import BorutaPy
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from scipy.stats import entropy
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFE, GenericUnivariateSelect, \
    RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from yellowbrick.classifier import confusion_matrix

from p1_eda.eda import get_details, check_duplicates, perform_correspondence_analysis
from p2_preprocessing.data_cleansing import impute_missing_values, perform_one_hot_encoding
from p2_preprocessing.feature_selection import get_feature_correlations
from p3_classification.baseline import get_baseline_performance
from p3_classification.upper_region_classifier import upper_region_classifier
from p4_hyper_parameter_tuning.rf_tuning import tune_random_forest
from p5_evaluation.model_evaluation import print_evaluation_results, plot_confusion_matrix

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

UR_CLASSES = ['2', '4', '10']
TR_CLASSES = ['1', '22']
IP_CLASSES = ['3', '5', '7', '11', '12', '13', '6']
EP_CLASSES = ['8', '14', '15', '17', '18', '19', '20', '16', '21']
TOP_LEVEL_TARGET = 'region'
SECOND_LEVEL_TARGET = 'class'

import matplotlib.pyplot as plt


def plot(x, y, predicted):
    plt.scatter(x, y, color='black')
    plt.plot(x, predicted, '-r')
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


def main_function(data_frame):
    get_details(data_frame)
    print("Class count\n", data_frame.groupby(SECOND_LEVEL_TARGET).size())

    # Impute missing values
    data_frame = impute_missing_values(data_frame, "most_frequent")
    print(data_frame.head(20))
    print(data_frame.isnull().sum().sum())



    # Top Level Classifier - classify by region
    # classify_by_region(pre_processed_data)

    # Check if duplicate records exist
    is_duplicated = check_duplicates(data_frame)
    # Drop duplicate records if exist
    if is_duplicated:
        data_frame.drop_duplicates(inplace=True)
        print("Dropped duplicate records. Size after dropping duplicates: ", data_frame.shape)

    # One Hot Encoding
    columns_to_encode = ['sex', 'histologic-type', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver',
                         'brain', 'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal']
    data_frame = perform_one_hot_encoding(data_frame, columns_to_encode)

    # Pre-prcoessed dataset
    pre_processed_data = data_frame

    # Top Level Classifier - classify by region
    # classify_by_region(pre_processed_data)

    # Create balanced datasets for the second level
    create_separate_datasets(pre_processed_data)
    #
    upper_region_classifier()

    thoracic_region_classifier()
    #
    ip_region_classifier()
    #
    ep_region_classifier()




# Creates balanced datasets for the second level sub classifiers
def create_separate_datasets(data_frame):
    # Remove all classes with only 1 instance
    class_filter = data_frame.groupby(SECOND_LEVEL_TARGET)
    data_frame = class_filter.filter(lambda x: len(x) > 1)
    print("2nd Level - Class Count", data_frame.groupby(SECOND_LEVEL_TARGET).size())

    # # Separate input feature & target variable
    X = data_frame.drop([SECOND_LEVEL_TARGET, TOP_LEVEL_TARGET], axis=1)  # drop class & region
    y = data_frame[SECOND_LEVEL_TARGET]  # Labels


    # # Split data into train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42, shuffle=True)


    ##########   Handle Class Imabalnce  #########
    sm = RandomOverSampler()
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    print("After Oversampling By Class\n", (pd.DataFrame(y_resampled)).groupby('class').size())

    # Convert to dataframes
    X_resampled = pd.DataFrame(X_resampled)
    y_resampled = pd.DataFrame(y_resampled)
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)

    # Train_dataset = Combine resamppled training data = X_resampled + y_resampled
    train_class_labels = y_resampled['class']
    train_dataset = X_resampled.join(train_class_labels)
    # train_dataset.to_csv('../resources/datasets/train.csv', index=False)

    # Test_dataset = Combine test data = X_test + y_test
    test_class_labels = y_test['class']
    test_dataset = X_test.join(test_class_labels)
    # test_dataset.to_csv('../resources/datasets/test.csv', index=False)

    # Filter out the instances according to their regions
    ur_train_set = train_dataset[train_dataset['class'].isin(UR_CLASSES)]
    ur_test_set = test_dataset[test_dataset['class'].isin(UR_CLASSES)]
    print(ur_train_set.shape)
    print(ur_train_set.groupby('class').size())
    print(ur_test_set.shape)
    ur_train_set.to_csv('../resources/datasets/ur_train_set.csv', index=False)
    ur_test_set.to_csv('../resources/datasets/ur_test_set.csv', index=False)


    tr_train_set = train_dataset[train_dataset['class'].isin(TR_CLASSES)]
    tr_test_set = test_dataset[test_dataset['class'].isin(TR_CLASSES)]
    print(tr_train_set.shape)
    print(tr_train_set.groupby('class').size())
    print(tr_test_set.shape)
    tr_train_set.to_csv('../resources/datasets/tr_train_set.csv', index=False)
    tr_test_set.to_csv('../resources/datasets/tr_test_set.csv', index=False)
    #
    #
    ip_train_set = train_dataset[train_dataset['class'].isin(IP_CLASSES)]
    ip_test_set = test_dataset[test_dataset['class'].isin(IP_CLASSES)]
    print(ip_train_set.shape)
    print(ip_train_set.groupby('class').size())
    print(ip_test_set.shape)
    ip_train_set.to_csv('../resources/datasets/ip_train_set.csv', index=False)
    ip_test_set.to_csv('../resources/datasets/ip_test_set.csv', index=False)
    #
    #
    ep_train_set = train_dataset[train_dataset['class'].isin(EP_CLASSES)]
    ep_test_set = test_dataset[test_dataset['class'].isin(EP_CLASSES)]
    print(ep_train_set.shape)
    print(ep_train_set.groupby('class').size())
    print(ep_test_set.shape)
    ep_train_set.to_csv('../resources/datasets/ep_train_set.csv', index=False)
    ep_test_set.to_csv('../resources/datasets/ep_test_set.csv', index=False)




def upper_region_classifier():
    ur_train_set = pd.read_csv("../resources/datasets/ur_train_set.csv", na_values='?', dtype='category')
    ur_test_set = pd.read_csv("../resources/datasets/ur_test_set.csv", na_values='?', dtype='category')

    # Separate training feature & training labels
    # X = ur_dataset.drop(['class'], axis=1)
    # y = ur_dataset['class']

    X_train = ur_train_set.drop(['class'], axis=1)
    y_train = ur_train_set['class']
    #
    # # Separate testing feature & testing labels
    X_test = ur_test_set.drop(['class'], axis=1)
    y_test = ur_test_set['class']

    # get_baseline_performance(X_train, y_train, X_test, y_test)
    #
    # spot_check_algorithms(X_train, y_train)

    model = RandomForestClassifier()
    model = model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print_evaluation_results(y_train, predictions)
    #
    predictions = model.predict(X_test)
    print_evaluation_results(y_test, predictions, train=False)

    joblib.dump(model, filename='../resources/models/ur_classifier.pkl')






def thoracic_region_classifier():
    tr_train_set = pd.read_csv("../resources/datasets/tr_train_set.csv", na_values='?', dtype='category')
    tr_test_set = pd.read_csv("../resources/datasets/tr_test_set.csv", na_values='?', dtype='category')

    # Separate training feature & training labels
    X_train = tr_train_set.drop(['class'], axis=1)
    y_train = tr_train_set['class']

    # Separate testing feature & testing labels
    X_test = tr_test_set.drop(['class'], axis=1)
    y_test = tr_test_set['class']

    get_baseline_performance(X_train, y_train, X_test, y_test)

    model = RandomForestClassifier()
    model = model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print_evaluation_results(y_train, predictions)

    predictions = model.predict(X_test)
    print_evaluation_results(y_test, predictions, train=False)

    joblib.dump(model, filename='../resources/models/tr_classifier.pkl')






def ip_region_classifier():
    ip_train_set = pd.read_csv("../resources/datasets/ip_train_set.csv", dtype='category')
    ip_test_set = pd.read_csv("../resources/datasets/ip_test_set.csv", dtype='category')

    # print("ip missing ", ip_train_set.isnull().sum().sum())
    # get_feature_correlations(ip_train_set)

    # Separate training feature & training labels
    X_train = ip_train_set.drop(['class'], axis=1)
    y_train = ip_train_set['class']

    # Separate testing feature & testing labels
    X_test = ip_test_set.drop(['class'], axis=1)
    y_test = ip_test_set['class']

    get_baseline_performance(X_train, y_train, X_test, y_test)

    model = RandomForestClassifier(random_state=42)
    model = model.fit(X_train, y_train)

    # https://towardsdatascience.com/machine-learning-kaggle-competition-part-two-improving-e5b4d61ab4b8


    # https://www.kaggle.com/residentmario/automated-feature-selection-with-sklearn
    # pd.Series(model.feature_importances_, index=X_train.columns[0:]).plot.bar(color='steelblue', figsize=(12, 6))
    # plt.show()


    # from sklearn.feature_selection import mutual_info_classif
    # kepler_mutual_information = mutual_info_classif(X_train, y_train)
    # plt.subplots(1, figsize=(26, 1))
    # sns.heatmap(kepler_mutual_information[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
    # plt.yticks([], [])
    # plt.gca().set_xticklabels(X_train.columns[0:], rotation=45, ha='right', fontsize=12)
    # plt.suptitle("Kepler Variable Importance (mutual_info_classif)", fontsize=18, y=1.2)
    # plt.gcf().subplots_adjust(wspace=0.2)
    # plt.show()
    #
    # trans = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=50)
    # kepler_X_trans = trans.fit_transform(X_train, y_train)
    # kepler_X_test_trans = trans.transform(X_test)
    # print("We started with {0} features but retained only {1} of them!".format(X_train.shape[1] - 1,
    #                                                                            kepler_X_trans.shape[1]))



    # https://www.kaggle.com/yaldazare/feature-selection-and-data-visualization
    # we will not only find best features but we also find how many features do we need for best accuracy.
    # The "accuracy" scoring is proportional to the number of correct classifications
    clf_rf_4 = RandomForestClassifier()
    # cv = KFold(n_repeats=3, n_splits=10, random_state=42)
    rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5, scoring='f1_micro')  # 5-fold cross-validation
    rfecv = rfecv.fit(X_train, y_train)

    print('Optimal number of features :', rfecv.n_features_)
    print('Best features :', X_train.columns[rfecv.support_])

    # Plot number of features VS. cross-validation scores
    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    clr_rf_5 = model.fit(X_train, y_train)
    importances = clr_rf_5.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure(1, figsize=(14, 13))
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    model = tune_random_forest(RandomForestClassifierX_train,  y_train)
    predictions = model.predict(X_train)
    print_evaluation_results(y_train, predictions)

    predictions = model.predict(X_test)
    print_evaluation_results(y_test, predictions, train=False)

    joblib.dump(model, filename='../resources/models/ip_classifier.pkl')







def ep_region_classifier():
    ep_train_set = pd.read_csv("../resources/datasets/ep_train_set.csv", na_values='?', dtype='category')
    ep_test_set = pd.read_csv("../resources/datasets/ep_test_set.csv", na_values='?', dtype='category')

    # Separate training feature & training labels
    X_train = ep_train_set.drop(['class'], axis=1)
    y_train = ep_train_set['class']

    # Separate testing feature & testing labels
    X_test = ep_test_set.drop(['class'], axis=1)
    y_test = ep_test_set['class']

    X_train, X_Val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1,
                                                      random_state=42, shuffle=True)

    model = RandomForestClassifier()

    ########################################### Hyper-parameter Tuning ##########################################
    # Perform grid search on the classifier using f1 score as the scoring method
    grid_obj = GridSearchCV(
        estimator=model,
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
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the best estimator
    best_clf = grid_fit.best_estimator_
    print(best_clf)

    predictions = best_clf.predict(X_Val)
    print_evaluation_results(y_val, predictions, train=False)

    model = best_clf

    ########################################### Final Model ###########################################
    model = model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print_evaluation_results(y_train, predictions)

    predictions = model.predict(X_test)
    print_evaluation_results(y_test, predictions, train=False)

    joblib.dump(model, filename='../resources/models/ep_classifier.pkl')


# Top Level Classifier
def classify_by_region(data_frame):
    X = data_frame.drop([TOP_LEVEL_TARGET, SECOND_LEVEL_TARGET], axis=1)  # Features - drop region, class
    y = data_frame[TOP_LEVEL_TARGET]  # Labels

    get_feature_correlations(data_frame, plot=True, return_resulst=False)
    # mutual_info = mutual_info_classif(X, y, discrete_features='auto')
    # print("mutual_info: ", mutual_info)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42, shuffle=True)

    ##########   Handle Class Imabalnce  #########
    sm = BorderlineSMOTE()
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    print("After Oversampling By Region\n", (pd.DataFrame(y_resampled)).groupby('region').size())

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

    # mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes = [100]*5, random_state=42))


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





    # pipeline = Pipeline(
    #         [
    #             # ('selector', SelectKBest(f_classif)),
    #             ('model',  RandomForestClassifier(n_jobs = -1) )
    #         ]
    # )
    #
    # # Perform grid search on the classifier using f1 score as the scoring method
    # grid_obj = GridSearchCV(
    #         estimator= GradientBoostingClassifier(),
    #         param_grid={
    #             # 'selector__k': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    #             'n_estimators': [10, 20, 30],
    #             'max_depth': [6, 10, 20, 30],
    #             # 'max_depth': [1, 10, 20, 30],
    #             'min_samples_split': [1, 10, 100]
    #             # 'model__n_estimators': np.arange(10, 200, 10)
    #             # 'C': [1, 10, 100]
    #         },
    #
    #         n_jobs=-1,
    #         scoring="f1_micro",
    #         cv=5,
    #         verbose=3
    # )
    #
    # # Fit the grid search object to the training data and find the optimal parameters
    # grid_fit =  grid_obj.fit(X_resampled, y_resampled)

    # # Get the best estimator
    # best_clf = grid_fit.best_estimator_
    # print(best_clf)


    # Get the final model
    parent_model = SVC(kernel='rbf',
                       C=10)  # KNN(n_neighbors = 7)-0.52 # LR(multiclass-ovr) -0.66, 0.67, 0.67, 0.69, 0.69, 0.68  MLP wid fs - 0.65, 0.69, 0.70,   GB - 0.67, without fs 0.62, 0.61,    DT - 0.58,   RF - 0.67,  multi_LR - wid fs 0.64 , voting - 0.60

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


data_frame1 = pd.read_csv("../resources/datasets/primary-tumor-with-region.csv", na_values='?', dtype='category')
# data_frame2 = pd.read_csv("../resources/datasets/synthetic_minority_samples.csv", na_values='?', dtype='category')
frames = [data_frame1]
data_frame = pd.concat(frames)

main_function(data_frame)







