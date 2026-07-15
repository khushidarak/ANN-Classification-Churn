# ANN-Classification-Churn

## Project Title
Customer Churn Prediction using Artificial Neural Networks (ANN)

## Problem Statement
Customer churn is a critical problem for businesses, especially in the banking and financial services sector. Predicting which customers are likely to leave (churn) enables companies to take proactive measures to retain them. This project builds a machine learning model using Artificial Neural Networks to predict customer churn based on various demographic and behavioral features.

## Dataset Used
The project uses a bank customer churn dataset that includes the following features:
- **CreditScore**: Customer's credit score
- **Geography**: Customer's location (France, Germany, Spain)
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: Number of years the customer has been with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of bank products the customer uses
- **HasCrCard**: Whether the customer has a credit card (0/1)
- **IsActiveMember**: Whether the customer is an active member (0/1)
- **EstimatedSalary**: Customer's estimated salary
- **Exited**: Target variable (whether the customer churned or not)

## Technologies
- **TensorFlow**: For building and training the neural network model
- **Streamlit**: For creating the interactive web application
- **Scikit-learn**: For data preprocessing, encoding, and scaling
- **Pandas**: For data manipulation
- **NumPy**: For numerical operations

## Model Architecture
The model is built using a sequential neural network with the following architecture:
- **Input Layer**: Accepts preprocessed features (12 input features after one-hot encoding)
- **Hidden Layers**: Multiple dense layers with ReLU activation functions
- **Output Layer**: Single neuron with sigmoid activation for binary classification
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

The model is trained on preprocessed data with:
- Label encoding for categorical variables (Gender)
- One-hot encoding for Geography
- Standard scaling for numerical features


## How to Run the Project Locally

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone the repository or navigate to the project directory:
```bash
cd /Users/khushidarak/Desktop/code/generative-ai-course/PROJECT ANN
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the following files are present in the directory:
- `model.h5` (trained model)
- `label_encoder_gender.pkl` (gender encoder)
- `onehot_encoder_geo.pkl` (geography encoder)
- `scaler.pkl` (feature scaler)

4. Run the Streamlit application:
```bash
streamlit run app.py
```

5. The application will open in your default web browser at `http://localhost:8501`

### Usage
- Select the customer's geography and gender
- Adjust the sliders for age, tenure, and number of products
- Enter numerical values for balance, credit score, and estimated salary
- Select binary options for credit card ownership and active member status
- Click to see the prediction result indicating whether the customer is likely to churn

## Deployment Link
https://ann-classification-churn-4xfnfh5pifcxi54pdql6t5.streamlit.app/

https://github.com/khushidarak/ANN-Classification-Churn



1. Dataset
        ↓
2. Data Preprocessing
        ↓
3. Train-Test Split
        ↓
4. Feature Scaling
        ↓
5. Build ANN Model
        ↓
6. Train the Model
        ↓
7. Save the Model & Preprocessing Objects
        ↓
8. Deploy with Streamlit
        ↓
9. User Enters Customer Details
        ↓
10. Preprocess User Input
        ↓
11. Predict Churn Probability
        ↓
12. Display Result