from dython import nominal
from dython.nominal import associations
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif



# Find the correlations between features
def get_feature_correlations(data_frame, plot=True, return_resulst=False):
    # source - https://github.com/shakedzy/dython/blob/master/dython/examples.py
    # https: // github.com / shakedzy / dython / issues / 2
    # associations(data_frame)
    return nominal.associations(data_frame, theil_u=True, plot=True, return_resulst=True)




# https://machinelearningmastery.com/feature-selection-with-categorical-data/
# feature selection - chi2
def select_featuresC(X_train, y_train, X_test, method):
	fs = SelectKBest(score_func=chi2, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs



# feature selection - mutual_info_classif
def select_featuresM(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs



# feature selection - mutual_info_classif
def select_features(X_train, y_train, X_test, method):
	fs = SelectKBest(score_func=method, k=15) # 13 - reduces f1, 15 - better
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs