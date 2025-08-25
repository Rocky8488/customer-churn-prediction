## Churn Prediction and Salary Regression Models

This repository contains two machine learning projects that use an Artificial Neural Network (ANN) to solve different problems based on the same dataset. The projects are implemented in Jupyter notebooks.

---

Classification.ipynb: Customer Churn Prediction

Project Description: This notebook details the process of building an ANN model to predict customer churn. It takes customer information, such as credit score, age, and location, and classifies whether a customer will exit the bank or not.

Dataset: The model is trained on the `Churn_Modelling.csv` dataset.

Technologies:
    Python
    Pandas
    Numpy
    Scikit-learn
    TensorFlow/Keras

Methodology:

    1.  Data Preprocessing: The notebook handles categorical features like 'Gender' and 'Geography' using `LabelEncoder` and `OneHotEncoder`, respectively. The `RowNumber`, `CustomerId`, and `Surname` columns are dropped as they are not relevant to the prediction. The data is then scaled using `StandardScaler` to ensure all features contribute equally to the model.
    2.  Model Architecture: A sequential ANN model is built with two dense hidden layers using 'relu' activation and a final output layer with a 'sigmoid' activation for binary classification.
    3.  Training: The model is compiled using the Adam optimizer and `BinaryCrossentropy` loss function, with 'accuracy' as the evaluation metric. It is trained for up to 100 epochs, incorporating `EarlyStopping` and `TensorBoard` callbacks to prevent overfitting and visualize the training process.

---

### Estimated_salary_regression.ipynb: Estimated Salary Regression

Project Description: This notebook focuses on building an ANN model to predict the estimated salary of customers based on their demographic and financial information. The model is built using the same dataset as the classification problem but with a different target variable.

Dataset: The model is trained on the `Churn_Modelling.csv` dataset.

Technologies:
    Python
    Pandas
    Numpy
    Scikit-learn
    TensorFlow/Keras

Methodology:

    1.  Data Preprocessing: Similar to the classification problem, this notebook preprocesses categorical features and scales the data using `LabelEncoder`, `OneHotEncoder`, and `StandardScaler`. The target variable, however, is `EstimatedSalary`.
    2.  Model Architecture: An ANN model is constructed with two dense hidden layers using 'relu' activation and a single-neuron output layer without an activation function, which is standard for regression tasks.
    3.  Training: The model is compiled with the Adam optimizer and `MeanAbsoluteError` loss function, with 'mae' as the evaluation metric. It is trained for 100 epochs with `EarlyStopping` and `TensorBoard` callbacks enabled.

---

### How to Run

1.  Ensure you have the required dependencies installed (e.g., `tensorflow`, `scikit-learn`, `pandas`, `numpy`).
2.  Place the `Churn_Modelling.csv` file in the same directory as the notebooks.
3.  Open and run the desired notebook in a Jupyter environment. The notebooks will save the trained models and scalers as `model.h5` and `*.pkl` files respectively.
