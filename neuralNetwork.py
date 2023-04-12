from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from evaluation import evaluate_nn
import numpy as np

# Train a neural network model
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

def train_nn_model(X_train, y_train, X_val, y_val, input_shape, num_classes, learning_rate=0.001, batch_size=32, epochs=10):
    """
    Trains Neural Network model
    :param X_train: Input features for training data
    :param y_train: Target labels for training data
    :param X_val: Input features for validation data
    :param y_val: Target labels for validation data
    :param input_shape: Shape of the input data
    :param num_classes: Number of classes for the target variable
    :param learning_rate: Learning rate for the optimizer (default is 0.001)
    :param batch_size: Number of samples per batch for training (default is 32)
    :param epochs: Number of epochs to train the model for (default is 10)
    :return: trained model, training history, and predictions on validation set
    """

    # Split the dataset into training, validation, and test sets
    X_test, y_test = X_val[:len(X_val)//2], y_val[:len(X_val)//2]
    X_val, y_val = X_val[len(X_val)//2:], y_val[len(X_val)//2:]

    # Create the model
    model = create_nn_model(input_shape, num_classes)

    # Compile the model
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Convert the target variable to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)

    # Make predictions on the validation set
    predictions = np.argmax(model.predict(X_val), axis=-1)

    # Evaluate the model on the test set
    y_test = to_categorical(y_test, num_classes)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test loss = {test_loss} and test accuracy = {test_acc}')
    evaluate_nn(predictions, history, y_test)
    # Return the trained model, training history, and predictions on validation set
    return model
