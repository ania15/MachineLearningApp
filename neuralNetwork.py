from keras.layers import Dropout
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from evaluation import evaluate_nn
import numpy as np
import tensorflow as tf


def find_best_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Creates Neural Network model
    :param input_shape: the initial input shape
    :param num_classes: number of classes
    :return best_params
    """
    best_score = 0
    best_params = {}
    for lr in [0.001, 0.01, 0.1]:
        for batch_size in [32, 64, 128]:
                print('im here')
                model = create_nn_model(X_train.shape[1:], num_classes=len(np.unique(y_train)))
                model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=10,
                                    verbose=0)
                score = history.history['val_accuracy'][-1]
                if score > best_score:
                    best_score = score
                    best_params = {'lr': lr, 'batch_size': batch_size}
    print(f'Best score: {best_score}')
    print(f'Best hyperparameters: {best_params}')
    return best_params


def plot_best_hyperparameters(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def create_nn_model(input_shape, num_classes):
    """
    Creates Neural Network model
    :param input_shape: the initial input shape
    :param num_classes: number of classes
    :return model: created model
    """
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train_nn_model(data):
    """
    Trains Neural Network model
    :param data: dataset on which the model will be trained
    :return: trained model
    """
    # Split the dataset into features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encode the labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # Define the neural network architecture
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(7, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)

    return model
