import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

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