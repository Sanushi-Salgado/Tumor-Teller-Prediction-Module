from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def tune_gb(X, y):
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
    grid_fit = grid_obj.fit(X, y)

    # Get the best estimator
    best_clf_rf = grid_fit.best_estimator_
    print(best_clf_rf)
    return best_clf_rf