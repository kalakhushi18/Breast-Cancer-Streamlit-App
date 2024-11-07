# Breast Cancer Cell Classification

## Overview

This project implements a machine learning model to classify breast cancer cells as benign or malignant. It uses logistic regression for classification and includes a Streamlit application for interactive prediction and visualization.

## Features

1. **Data Processing**: Cleans and prepares breast cancer cell data.
2. **Machine Learning Model**: Utilizes logistic regression for classification.
3. **Interactive Web Application**: Built with Streamlit for real-time predictions.
4. **Data Visualization**: Uses Plotly to create an interactive radar chart of cell features.
5. **Parameter Tuning**: Allows users to adjust cell nuclei parameters via sliders.

## Requirements

- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn (for model training, not shown in the provided code)
- Pickle (for model serialization)


## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Use the sidebar sliders to adjust cell nuclei parameters.
3. View the radar chart showing scaled feature values.
4. Check the prediction result (Benign or Malignant) and probabilities.

## Project Structure

- `app.py`: Main Streamlit application file.
- `data/data.csv`: Dataset containing breast cancer cell features.
- `model/model.pkl`: Serialized logistic regression model.
- `model/scaler.pkl`: Serialized scaler for feature normalization.

## Key Components

1. **Data Cleaning**: Removes duplicates and unnecessary columns from the dataset.
2. **Feature Scaling**: Normalizes input features for consistent model performance.
3. **Interactive Sidebar**: Allows users to adjust 30 different cell nuclei parameters.
4. **Radar Chart**: Visualizes mean, standard error, and worst values of cell features.
5. **Prediction**: Uses a pre-trained logistic regression model to classify cells.
6. **Result Display**: Shows classification result and probabilities.

## Model Details

- The model uses logistic regression 
- Input features are scaled using a standard scaler before prediction.
- The model predicts whether a cell cluster is benign or malignant.

## Future Enhancements

- Implement model retraining within the app.
- Add more visualization options for better data interpretation.
- Incorporate additional machine learning models for comparison.


## Author

Khushi Kala

---

