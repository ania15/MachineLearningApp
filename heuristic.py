from random import random
from sklearn.model_selection import train_test_split
from evaluation import evaluate


def heuristic_clf(data):
    """
    Performs a simple heuristic that classifies the data.
    It is not accurate, based only on author's observations.
    :param data: data to be classified
    :param cover_type: predicted cover_type
    """
    cover_type = []
    for index, row in data.iterrows():
        if row['Elevation'] > 3000:
            cover_type.append(1)
        elif row['Elevation'] < 2800:
            cover_type.append(5)
        elif 2800 <= row['Elevation'] <= 3000:
            cover_type.append(2)
        else:
            numbers = [3, 4, 6, 7]
            cover_type.append(numbers[random.randint(0, len(numbers)-1)])
    return cover_type


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

