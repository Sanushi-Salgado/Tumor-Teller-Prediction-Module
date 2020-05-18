import itertools
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from IPython.core.pylabtools import figsize
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report, accuracy_score, precision_score, \
    recall_score, hamming_loss, jaccard_similarity_score, confusion_matrix
import numpy as np





def print_evaluation_results(y_test, predictions, train=True):
    print("\n!!!!!!!!!!!!!!!!!!!!! EVALUATION RESULTS !!!!!!!!!!!!!!!!!!!!!!")
    if train:
        print("---------- Training Performance ----------")
    else:
        print("---------- Testing Performance ----------")
    print("Accuracy Score ", accuracy_score(y_test, predictions))
    print("Hamming Loss ", hamming_loss(y_test, predictions))
    print("Jaccard Similarity Score ", jaccard_similarity_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
    print(classification_report_imbalanced(y_test, predictions))
    print()


 # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, predictions, labels=['11', '12', '13', '3', '5', '7'])
    # sns.heatmap(cm, annot=True, fmt="d")
    # plt.show()

# https://github.com/javaidnabi31/Multi-class-with-imbalanced-dataset-classification/blob/master/20-news-group-classification.ipynb
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
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()





def get_scores(actual_values, predictions, test=False):
        print( "Accuracy: ", accuracy_score(actual_values, predictions) )
        print( "Error: ", 1 - accuracy_score(actual_values, predictions) )

        print( 'Precision macro is:' + str(round(precision_score(actual_values, predictions, average='macro'), 2)))
        print( 'Precision micro is:' + str(round(precision_score(actual_values, predictions, average='micro'), 2)))

        print( 'Recall macro is:' + str(round(recall_score(actual_values, predictions, average='macro'), 2)))
        print( 'Recall micro is:' + str(round(recall_score(actual_values, predictions, average='micro'), 2)))

        print( 'F1 macro:', str(round(f1_score(actual_values, predictions, average='macro'), 2)))
        print( 'F1 weighted:', str(round(f1_score(actual_values, predictions, average='weighted'), 2)))
        print( 'F1 micro:', str(round(f1_score(actual_values, predictions, average='micro'), 2)))

        print( 'Kappa', cohen_kappa_score(actual_values, predictions))

        print(classification_report(actual_values, predictions))
        print(classification_report_imbalanced(actual_values, predictions))

        # print( 'ROC area', metrics.roc_curve(y, predictions, pos_label="1") )

        # plot_confusion_matrix(cnf_matrix, classes=np.asarray(label_names), normalize=True,
                              # title='Normalized confusion matrix')

        # cm = confusion_matrix(actual_values, predictions)
        # print( "Confusion Matrix: \n", cm)
        # plt.subplots(figsize=(12, 9))
        # sns.heatmap(cm, annot=True)
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.show()


        # fpr, tpr, thres = roc_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label='1')
        # graph.figure(figsize=(4, 4))
        # graph.plot(fpr, tpr, label='Test')
        # graph.xlabel('FPR')
        # graph.ylabel('TPR')
        # graph.show()






def evaluate_models(models_to_evaluate, training_features, training_labels, testing_features, testing_labels, title, x_label, y_label):
    # Takes in a model, trains the model & evaluates the model performance on both training & test sets
    for key in models_to_evaluate:
        model = models_to_evaluate[key]
        f1_scores_on_test_set = []
        kappa_scores_on_test_set = []

        start_time = time()
        # Train the model, on training datasets
        model = model.fit(training_features, training_labels)
        end_time = time()

        print("#############  %s  #############" % key)
        print("---------  Training Performance  --------")
        print("Time taken to train the model (Train time):", round(end_time - start_time, 3), "s")  # the time would be round to 3 decimal in seconds
        start_time = time()
        # Make predictions on training datasets & evaluate
        predictions = model.predict(training_features)
        end_time = time()
        print("Time taken to generate predictions on training set (Prediction Time):", round(end_time - start_time, 3), "s")
        get_scores(training_labels, predictions)


        print("---------  Testing Performance  --------")
        start_time = time()
        # Make predictions on test datasets or unseen datasets & evaluate
        predictions = model.predict(testing_features)
        end_time = time()
        print("Time taken to generate predictions on test set (Prediction Time):", round(end_time - start_time, 3), "s")
        get_scores(testing_labels, predictions)

        f_score = round(f1_score(testing_labels, predictions, average='micro'), 2)
        f1_scores_on_test_set.append(f_score)
        kappa_score = round(cohen_kappa_score(testing_labels, predictions), 2)
        kappa_scores_on_test_set.append(kappa_score)


# # Dataframe to hold the results
# model_comparison = pd.DataFrame({'model': ['Deciion Tree', 'Random Forest', 'Gradient Boosting', 'Extra Trees'],
#                                  'f1': f1_scores_on_test_set
#                                  })
#
# # Horizontal bar chart of test mae# Horizontal bar chart of test mae
# model_comparison.sort_values('model', ascending=False).plot(x='model', y='f1', kind='bar', color='green',
#                                                             edgecolor='green')
#
# # Plot formatting
# plt.style.use('fivethirtyeight')
# figsize(8, 12)
# plt.title("F-Measure")
# # plt.yticks(size=14)
# plt.xlabel('Classification Models')
# # plt.xticks(size=14)
# # plt.title('Model Comparison on Test set using F-Measure', size=20)
# plt.show()






#
# def visualize_results(title, x_label, y_label):
#     # Dataframe to hold the results
#     model_comparison = pd.DataFrame({'model': ['Deciion Tree', 'Random Forest', 'Gradient Boosting', 'Extra Trees'],
#                                      'f1': f1_scores_on_test_set
#                                      })
#
#     # Horizontal bar chart of test mae# Horizontal bar chart of test mae
#     model_comparison.sort_values('model', ascending=False).plot(x='model', y='f1', kind='bar', color='green',
#                                                                 edgecolor='green')
#
#     # Plot formatting
#     plt.style.use('fivethirtyeight')
#     figsize(8, 12)
#     plt.title(title)
#     # plt.yticks(size=14)
#     plt.xlabel(x_label)
#     plt.xlabel(y_label)
#     # plt.xticks(size=14)
#     # plt.title('Model Comparison on Test set using F-Measure', size=20)
#     plt.show()
#
#
