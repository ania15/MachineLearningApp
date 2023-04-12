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

# Load the Covertype dataset
data = pd.read_csv('covtype.data', header=None)

# Add column names to the dataset
column_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points'] + [f'Wilderness_Area_{i}' for i in range(1, 5)] \
               + [f'Soil_Type_{i}' for i in range(1, 41)] + ['Cover_Type']
data.columns = column_names

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Normalize the training and testing data
train_norm = (train_data - train_data.mean()) / train_data.std()
test_norm = (test_data - train_data.mean()) / train_data.std()

# Extract the features and labels
X_train = train_norm.drop('Cover_Type', axis=1).values
y_train = train_norm['Cover_Type'].values
X_test = test_norm.drop('Cover_Type', axis=1).values
y_test = test_norm['Cover_Type'].values

# Preparing the models
dt_model = dt_clf(data)
lr_model = lr_clf(data)
# nn_model = train_nn_model(X_train, y_train, X_test, y_test, X_train.shape[1:])


@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.args.get('model')
    body = request.json
    body = pd.DataFrame([body])
    if selected_model == 'Heuristic':
        pass
        #means = data.groupby('Cover_Type').mean().values
        #return heuristic_clf(body, means)
    elif selected_model == 'Logistic Regression':
        return str(lr_model.predict(body))
    elif selected_model == 'Decision Tree':
        return str(dt_model.predict(body))
    else:
        pass
        #return "NN prediction" + str(body["Slope"])
