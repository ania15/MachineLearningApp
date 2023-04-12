def heuristic_clf(data):
    """
    Performs heuristic classification on dataset.
    :param data: dataset
    :return: classified data
    """
    majority_class = data[54].value_counts().idxmax()
    data['predicted_class'] = majority_class
    accuracy = (data[54] == data['predicted_class']).mean()
    print('Accuracy:', accuracy)
    return data['predicted_class'].unique()