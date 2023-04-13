import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request
from heuristic import heuristic_clf, evaluate_heu
from logisticRegression import lr_clf
from decisionTree import dt_clf
from neuralNetwork import train_nn_model

app = Flask(__name__)

# Loading the dataset and changing column names
data = pd.read_csv('covtype.data', header=None)
cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points'] + [f'Wilderness_Area_{i}' for i in range(1, 5)] \
               + [f'Soil_Type_{i}' for i in range(1, 41)] + ['Cover_Type']
data.columns = cols


# Preparing and evaluating the models
# dt_model = dt_clf(data)
#lr_model = lr_clf(data)
nn_model = train_nn_model(data)
evaluate_heu(data)


@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.args.get('model')
    body = request.json
    body = pd.DataFrame([body])
    if selected_model == 'Heuristic':
        return str(heuristic_clf(body))
    elif selected_model == 'Logistic Regression':
        return str(lr_model.predict(body))
    elif selected_model == 'Decision Tree':
        return str(dt_model.predict(body))
    elif selected_model == 'Neural Network':
        pred = nn_model.predict(body)
        pred = np.argmax(pred)
        return str(pred)
    else:
        return "The model you chose does not exist."

