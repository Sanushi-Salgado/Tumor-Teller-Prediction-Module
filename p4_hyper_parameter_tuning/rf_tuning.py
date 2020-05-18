from sklearn.model_selection import GridSearchCV


def tune_random_forest(model, X, y):
    # Perform grid search on the classifier using f1 score as the scoring method
    grid_obj = GridSearchCV(
        estimator=model,
        param_grid={
            'n_estimators': [10, 20, 30, 35, 40, 45, 50, 60],
            'max_depth': [4, 6, 8, 10, 15, 20, 25, 30, 50, 60],
            'min_samples_split': [10, 50, 100],
            'max_features': [9, 10, 12, 15, 17, 18],
            'bootstrap': [True, False],
            'criterion': ['gini'],

            # 'bootstrap': [True],
            # 'max_depth': [80, 90, 100, 110],
            # 'max_features': [2, 3],
            # 'min_samples_leaf': [3, 4, 5],
            # 'min_samples_split': [8, 10, 12],

            # 'model__n_estimators': np.arange(10, 200, 10)
            # 'C': [1, 10, 100]
            # 'min_samples_leaf' : [1]
        },
        n_jobs=-1,
        scoring="f1_micro",
        cv=5,
        verbose=1
    )
    # Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(X, y)

    # Get the best estimator
    best_clf_rf = grid_fit.best_estimator_
    print(best_clf_rf)
    return best_clf_rf