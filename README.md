# MachineLearningApp

## General Information

This app contains 4 Machine Learning models based on the algorithms listed below:
- Simple heuristic search
- Decision Tree
- Logistic Regression
- Neural Networks.

The data that the models have been trained on can be found here: https://archive.ics.uci.edu/ml/datasets/Covertype.
The target variable is Cover_type and the rest of the variables are features used to predict it.

Models are evaluated thorugh some standard metrics (Accuracy, Classification Report, Confusion Matrix, Loss) accordingly to their characteristics.

There is a REST API serving the models created with Flask.

All of the dependencies can be found in the requirements.txt file.

# Usage

User can choose a model and get a prediction by HTTP request. In order to that it is necessary to deliver input features:

- model type (Heuristic, Decision Tree, Logistic Regression, Neural Network)
- data with all the input features.

Below an example of usage by Postman App can be found:
![image](https://user-images.githubusercontent.com/74561797/231743795-115b5989-02fb-4dfd-b4a1-b7226a14f02e.png)
![image](https://user-images.githubusercontent.com/74561797/231743912-1ae27074-4946-40f3-a997-ee9c70a62d96.png)
You can find the body in the sample.json file. After clicking **SEND** the prediction appears below:
![image](https://user-images.githubusercontent.com/74561797/231744130-0aeea8b0-7e8f-4ea4-be4d-73468a7175df.png)

However, the HTTP Request can be done any other way preferred by user.

