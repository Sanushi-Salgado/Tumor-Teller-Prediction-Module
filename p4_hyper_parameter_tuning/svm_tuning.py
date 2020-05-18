from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def tune_SVM(X, y):
    params_grid = [{'kernel': ['rbf'],
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    grid_obj = GridSearchCV(
        estimator=SVC(),
        param_grid=params_grid,
        n_jobs=-1,
        scoring="f1_micro",
        cv=5,
        verbose=3
    )
    grid_fit = grid_obj.fit(X, y)

    # Get the best estimator
    best_clf_rf = grid_fit.best_estimator_
    print(best_clf_rf)
    return best_clf_rf