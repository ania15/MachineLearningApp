from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from evaluation import evaluate

def dt_clf(data):
    """
    Creates and trains a baseline model based on Decision Tree algorithm.
    This algorithm does not need data scaling/normalizing.
    :param data: dataset on which the model will be trained
    :return dt_model: trained model
    """
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.25,
                                                        random_state=42)
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    evaluate(dt_preds, y_test)
    return dt_model
