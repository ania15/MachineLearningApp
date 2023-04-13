from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from heuristic import heuristic_clf


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


def evaluate_heu(data):
    """
    Evaluates the performance of a simple heuristic on the whole dataset.
    The data is not normalized because the Cover_type is predicted (1-7).
    :param data: dataset on which the heuristic will be eavluated
    """
    # Extracting only test data for the evaluation purposes
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    X_test = test_data.drop('Cover_Type', axis=1)
    y_test = test_data['Cover_Type']

    # Applying the algorithm on test set and evaluating
    X_test['Heuristic_Cover_Type'] = heuristic_clf(X_test)
    evaluate(X_test['Heuristic_Cover_Type'], y_test)
