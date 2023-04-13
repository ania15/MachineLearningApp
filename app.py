import numpy as np
import pandas as pd
from flask import Flask, request
from heuristic import heuristic_clf, evaluate_heu
from logisticRegression import lr_clf
from decisionTree import dt_clf
from neuralNetwork import train_nn_model


app = Flask(__name__)

# Loading the dataset and changing column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
data = pd.read_csv(url, header=None)
cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points'] + [f'Wilderness_Area_{i}' for i in range(1, 5)] \
               + [f'Soil_Type_{i}' for i in range(1, 41)] + ['Cover_Type']
data.columns = cols

# Preparing and evaluating the models
print('Heuristic Search model evaluation')
evaluate_heu(data)
dt_model = dt_clf(data)
lr_model = lr_clf(data)
nn_model = train_nn_model(data)

# REST API
@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.args.get('model')
    body = request.json
    body = pd.DataFrame([body])
    if selected_model == 'Heuristic':
        pred = heuristic_clf(body)
        return str(pred)
    elif selected_model == 'Logistic Regression':
        pred = lr_model.predict(body)
        return str(pred)
    elif selected_model == 'Decision Tree':
        pred = dt_model.predict(body)
        return str(pred)
    elif selected_model == 'Neural Network':
        pred = nn_model.predict(body)
        pred = np.argmax(pred) + 1
        return str(pred)
    else:
        return "The model you chose does not exist."

