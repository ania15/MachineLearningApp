from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def build_model(input_shape, num_classes, dropout_rate=0.2, activation="relu", optimizer="adam"):
    """
    Builds a Keras neural network model with the given hyperparameters.

    :param input_shape: Shape of the input data.
    :param num_classes: Number of classes in the target labels.
    :param dropout_rate: optional (default=0.2) Dropout rate to use between layers.
    :param activation: optional (default='relu') Activation function to use in the hidden layers.
    :param optimizer: optional (default='adam') Optimizer to use during training.
    :return: a Keras Sequential model object.
    """

    # Defining the layers
    model = Sequential([
        Dense(128, activation=activation, input_shape=input_shape),
        Dropout(dropout_rate),
        Dense(64, activation=activation),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax")
    ])

    # Compiling the model
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def find_best_hyperparams(X, y, num_classes, n_iter=4, cv=3, epochs=6, batch_size=128):
    """
    Finds the best hyperparameters for a Keras neural network model using random search.
    :param X: Input data
    :param y: Target labels.
    :param num_classes: Number of classes in the target labels.
    :param n_iter: optional (default=5) Number of parameter settings that are sampled.
    :param cv: optional (default=2) Determines the cross-validation splitting strategy.
    :param epochs: int, optional (default=10) Number of epochs to train the model.
    :param batch_size: optional (default=128) Batch size used during training.
    :return: dictionary containing the best hyperparameters found by random search.
    """
    input_shape = X.shape[1:]

    # Defining the hyperparameters
    param_distribs = {
        "dropout_rate": [0.1, 0.2, 0.3],
        "activation": ["relu", "elu"],
        "optimizer": ["adam", "rmsprop"]
    }

    # Creating a classifier and performing randomized search
    keras_clf = KerasClassifier(build_fn=build_model, input_shape=input_shape, num_classes=num_classes, epochs=epochs, batch_size=batch_size, verbose=0)
    random_search = RandomizedSearchCV(keras_clf, param_distributions=param_distribs, n_iter=n_iter, cv=cv, verbose=2)
    random_search.fit(X, y)
    return random_search.best_params_


def plot_best_hyperparameters(history):
    """
    Plots the best neural network parameters
    :param history: dictionary containing information about the training process of the model
    """

    # Plotting accuracy and loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set(title='Model Accuracy', ylabel='Accuracy', xlabel='Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper right')
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set(title='Model Loss', ylabel='Loss', xlabel='Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def train_nn_model(data):
    """
    Trains Neural Network model
    :param data: dataset on which the model will be trained
    :return: trained model
    """
    # Splitting data into training and test sets, scaling the features and encoding labels
    X =data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # Finding best parameters on a sample set - the original set is too big
    idx = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.01), replace=False)
    X_sample = X[idx]
    y_sample = y[idx]
    best_params = find_best_hyperparams(X_sample, y_sample, num_classes=7)

    # Building and training the model on the full dataset based on the best hyperparameters
    # with early stopping
    model = build_model(X_train.shape[1:], num_classes=7, **best_params)
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=6, batch_size=128, validation_data=(X_test, y_test),
              callbacks=[early_stopping])
    history = model.fit(X_train, y_train, epochs=6, batch_size=128, validation_data=(X_test, y_test))

    # Evaluating the model
    print('Neural Network model evaluation')
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)
    plot_best_hyperparameters(history)
    return model
