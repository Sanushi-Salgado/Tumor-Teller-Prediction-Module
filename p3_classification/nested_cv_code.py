# # https://chrisalbon.com/machine_learning/model_evaluation/nested_cross_validation/
#         # Create Inner Cross Validation (For Parameter Tuning)
#         # Create a list of 10 candidate values for the C parameter
#         C_candidates = dict( C = np.logspace(-4, 4, 10))
#
#         # Create a gridsearch object with the support vector classifier and the C value candidates
#         clf = GridSearchCV(estimator=SVC(), param_grid=C_candidates)
#         # Fit the cross validated grid search on the data
#         clf.fit(X, y)
#
#         # Show the best value for C
#         print(clf.best_estimator_.C)
#
#         # Create Outer Cross Validation (For Model Evaluation)
#         cv_scores = cross_val_score(clf, X, y, scoring='f1_micro')
#         print("Mean Score: ", cv_scores.mean())






#  https://mlfromscratch.com/nested-cross-validation-python-code/#/
#         models_to_run = [RandomForestClassifier()]
#         # 1st param grid, corresponding to RandomForestRegressor
#         models_param_grid = [
#             {
#                 'max_depth': [3, None],
#                 'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#                 'max_features': [8, 10, 12, 15, 18]
#             }
#         ]
#
#         for i, model in enumerate(models_to_run):
#             nested_CV_search = NestedCV(model=model, params_grid=models_param_grid[i],
#                                         outer_kfolds=5, inner_kfolds=5,
#                                         cv_options={'sqrt_of_score': True, 'randomized_search_iter': 30})
#
#             nested_CV_search.fit(X=X, y=y)
#             model_param_grid = nested_CV_search.best_params
#
#             print(np.mean(nested_CV_search.outer_scores))
#             print(nested_CV_search.best_inner_params_list)