import numpy as np

def heuristic_clf(data, means):
    """
    Performs heuristic classification on dataset.
    :param data: dataset without Cover_Type column
    :param means: array of mean values for each cover type
    :return: predicted cover types
    """
    distances = np.sqrt(np.sum((data.values[:, :] - means[:, :]) ** 2, axis=1))
    predicted_classes = np.argmin(distances, axis=0) + 1
    return predicted_classes
