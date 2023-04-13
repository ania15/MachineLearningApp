from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from evaluation import evaluate


def lr_clf(data):
    """
    Creates and trains a baseline model based on Logistic Regression algorithm.
    The data is scaled at the beginning so the model works properly.
    :param data: dataset on which the model will be trained
    :return lr_model: trained model
    """
    # Splitting the data into training and test set, scaling the data
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.25,
                                                        random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Performing logistic regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Evaluating the model
    lr_preds = lr_model.predict(X_test_scaled)
    print('Logistic Regression model evaluation')
    evaluate(lr_preds, y_test)

    return lr_model
