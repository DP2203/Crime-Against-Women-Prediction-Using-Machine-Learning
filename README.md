# Crime Against Women Prediction Using Machine Learning

## Project Overview

This project is a **Machine Learning-based web application** built using
**Streamlit** that predicts crimes against women using historical crime
data.

The application allows users to: - Upload a dataset - Train multiple
machine learning models - Analyze crime trends - Predict future crime
counts

The system also includes **user authentication** and an **admin panel**
to manage users.

------------------------------------------------------------------------

## Features

### User Features

-   User registration and login
-   Upload crime dataset (CSV)
-   Select input features and target variables
-   Train machine learning models
-   View model performance metrics
-   Visualize actual vs predicted values
-   Predict crime counts for future years

### Admin Features

-   Admin login
-   View all registered users
-   Delete users
-   Manage system users

------------------------------------------------------------------------

## Machine Learning Models Used

-   Linear Regression
-   Random Forest Regressor
-   Decision Tree Regressor
-   Gradient Boosting Regressor

------------------------------------------------------------------------

## Technologies Used

  Technology     Purpose
  -------------- ------------------------------
  Python         Programming Language
  Streamlit      Web Application Framework
  Pandas         Data Analysis
  NumPy          Numerical Computation
  Matplotlib     Data Visualization
  Scikit-learn   Machine Learning
  SQLite         User Authentication Database

------------------------------------------------------------------------

## Project Structure

    crime-against-women-prediction-using-machine-learning
    │
    ├── webapp.py
    ├── requirements.txt
    ├── users.db
    ├── Fevicon.png
    └── README.md

------------------------------------------------------------------------

## Installation

### Clone the Repository

    git clone https://github.com/your-username/crime-against-women-prediction-using-machine-learning.git

### Navigate to Project Folder

    cd crime-against-women-prediction-using-machine-learning

### Install Dependencies

    pip install -r requirements.txt

### Run the Application

    streamlit run webapp.py

------------------------------------------------------------------------

## Requirements

Example requirements.txt

    streamlit==1.45.1
    pandas==2.2.3
    numpy==2.1.3
    matplotlib==3.8.4
    scikit-learn==1.6.1

------------------------------------------------------------------------

## Future Improvements

-   Crime analytics dashboard
-   Prediction history tracking
-   Improved data visualizations
-   Enhanced user interface

------------------------------------------------------------------------

## Author

Divy Patel\
BE Computer Engineering\
Sigma Institute of Engineering
