import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

app = Flask(__name__)

data = pd.read_csv('covtype.data', header=None)
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.25,
                                                        random_state=42)
input_shape = (X_train.shape[1],)
X_train_norm = StandardScaler().fit_transform(X_train)
X_test_norm = StandardScaler().fit_transform(X_test)

def evaluate(y_pred):
    """
    Evaluates parameters of the model
    :param y_pred: prediction to be evaluated
    """
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")


def evaluate_nn(y_pred, history):
    """
    Evaluates parameters of Neural Network model
    :param y_pred: prediction to be evaluated
    :param history: dictionary containing information about the training process of the model
    """
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Plot the training and validation curves
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Curves')
    plt.legend()
    plt.show()

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
    evaluate(dt_preds)
    return dt_preds


def lr_clf(data):
    """
    Creates and trains a baseline model based on Logistic Regression algorithm.
    :param data: dataset
    :return lr_preds: predictions
    """""
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.25,
                                                        random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    evaluate(lr_preds)
    return lr_preds

# Train a neural network model
def create_nn_model(input_shape):
    """
    Creates Neural Network model
    :param input_shape: the initial input shape
    :return model: created model
    """
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_nn_model(X_train, y_train, X_val, y_val, input_shape, learning_rate=0.001, batch_size=32, epochs=10):
    """
    Trains Neural Network model
    :param X_train: Input features for training data
    :param y_train: Target labels for training data
    :param X_val: Input features for validation data
    :param y_val: Target labels for validation data
    :param input_shape: Shape of the input data
    :param learning_rate: Learning rate for the optimizer (default is 0.001)
    :param batch_size: Number of samples per batch for training (default is 32)
    :param epochs: Number of epochs to train the model for (default is 10)
    :return: trained model, training history, and predictions on validation set
    """

    # Split the dataset into training, validation, and test sets
    X_test, y_test = X_val[:len(X_val)//2], y_val[:len(X_val)//2]
    X_val, y_val = X_val[len(X_val)//2:], y_val[len(X_val)//2:]

    # Create the model
    model = create_nn_model(input_shape)

    # Compile the model
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)

    # Make predictions on the validation set
    predictions = np.argmax(model.predict(X_val), axis=-1)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test loss = {test_loss} and test accuracy = {test_acc}')
    evaluate_nn(predictions, history)
    # Return the trained model, training history, and predictions on validation set
    return model, history, predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_model = request.form['model']
        if selected_model == 'Heuristic':
            prediction = heuristic_clf(data)
            prediction = np.array2string(prediction)
        elif selected_model == 'Logistic Regression':
            prediction = lr_clf(data)
            prediction = np.array2string(prediction)
        elif selected_model == 'Decision Tree':
            prediction = dt_clf(data)
            prediction = np.array2string(prediction)
        elif selected_model == 'Neural Network':
            model, history, prediction = train_nn_model(X_train_norm, y_train, X_test_norm, y_test, input_shape)
            prediction = np.array2string(prediction)
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html')
