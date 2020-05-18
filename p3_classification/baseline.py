# https://www.geeksforgeeks.org/ml-dummy-classifiers-using-sklearn/?ref=rp

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def get_baseline_performance(X_train, y_train, X_test, y_test):
        dummy = None
        for strategy in ['stratified', 'most_frequent', 'prior', 'uniform']:
                # if strategy == 'constant':
                #         dummy = DummyClassifier(strategy=strategy, constant='2', random_state=7)
                # else:
                dummy = DummyClassifier(strategy=strategy, constant=None, random_state=7)

                # Train the dummy model
                dummy.fit(X_train, y_train)
                print( '-' * 5 + '%s' % (strategy) )

                # Get baseline in performance, on the test set
                predictions = dummy.predict(X_test)
                print('-> F1 score:', f1_score(y_test, predictions, average='micro'))


