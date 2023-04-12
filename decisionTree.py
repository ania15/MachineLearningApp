from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from evaluation import evaluate

def dt_clf(data):
    """
    Creates and trains a baseline model based on Decision Tree algorithm.
    :param data: dataset
    :return dt_preds: predictions
    """
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.25,
                                                        random_state=42)
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    evaluate(dt_preds, y_test)
    return dt_model
