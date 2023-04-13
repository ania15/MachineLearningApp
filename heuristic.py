from random import random


def heuristic_clf(data):
    """
    Performs a simple heuristic that classifies the data.
    It is not accurate, based only on author's observations.
    :param data: data to be classified
    :param cover_type: predicted cover_type
    """

    # Performing a simple classification based on Elevation value
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
            cover_type.append(numbers[random.randint(0, len(numbers) - 1)])
    return cover_type
