import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
from heuristic import heuristic_clf
from logisticRegression import lr_clf
from decisionTree import dt_clf
from neuralNetwork import train_nn_model

app = Flask(__name__)

data = pd.read_csv('covtype.data', header=None)
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.25,
                                                    random_state=42)
input_shape = (X_train.shape[1],)
X_train_norm = StandardScaler().fit_transform(X_train)
X_test_norm = StandardScaler().fit_transform(X_test)


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
