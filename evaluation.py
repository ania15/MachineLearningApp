from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


def evaluate(y_pred, y_test):
    """
    Evaluates parameters of the model
    :param y_pred: prediction to be evaluated
    :param y_test: the target output
    """

    # Calculating basic parameters of the model in order to have a comparison between the models
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")



