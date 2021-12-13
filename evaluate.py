import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from EvoMSA.utils import bootstrap_confidence_interval
from sklearn.metrics import recall_score


def model_Evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)
    # incase the prediction dataformat is different from test format
    y_pred = number_string_convert(y_pred, 'str')

    print(classification_report(y_test, y_pred))

    # confidence = bootstrap_confidence_interval(
    #     np.array(y_test), np.array(y_pred), metric=lambda y_test, y_pred: recall_score(y_test, y_pred, average=None)[0])

    confidence = bootstrap_confidence_interval(np.array(y_test), np.array(y_pred),
                                               metric=lambda y, hy: (y == hy).mean())
    print(confidence)

    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['0', '4']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(
        value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)


def number_string_convert(y_predict, type):
    """ change data type of the label from number to string

    Args:

        y_predict (list()): a list of labels
    """
    if type == "str":
        return [str(label) for label in y_predict]

    if type == 'int':
        return [int(label) for label in y_predict]
