#importing required libaries 

import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import pickle
import numpy as np

#reading  data

def get_cleaned_data():
   data = pd.read_csv('..\data\data.csv')

   #dropping duplicates
   data = data.drop_duplicates()

   #dropping unnecessary col
   data = data.drop(columns=['Unnamed: 32', 'id'])

   return data

# for radar chart required
def get_scaled_values(input_dict):
  data = get_cleaned_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

#desinging the side bar
def add_sidebar():
    st.sidebar.header("Cell Nuclei Details")
    data = get_cleaned_data()

    slider_labels = [
  ('Radius_mean', 'radius_mean'),
  ('Texture_mean', 'texture_mean'),
  ('Perimeter_mean', 'perimeter_mean'),
  ('Area_mean', 'area_mean'),
  ('Smoothness_mean', 'smoothness_mean'),
  ('Compactness_mean', 'compactness_mean'),
  ('Concavity_mean', 'concavity_mean'),
  ('Concave points_mean', 'concave points_mean'),
  ('Symmetry_mean', 'symmetry_mean'),
  ('Fractal_dimension_mean', 'fractal_dimension_mean'),
  ('Radius_se', 'radius_se'),
  ('Texture_se', 'texture_se'),
  ('Perimeter_se', 'perimeter_se'),
  ('Area_se', 'area_se'),
  ('Smoothness_se', 'smoothness_se'),
  ('Compactness_se', 'compactness_se'),
  ('Concavity_se', 'concavity_se'),
  ('Concave points_se', 'concave points_se'),
  ('Symmetry_se', 'symmetry_se'),
  ('Fractal_dimension_se', 'fractal_dimension_se'),
  ('Radius_worst', 'radius_worst'),
  ('Texture_worst', 'texture_worst'),
  ('Perimeter_worst', 'perimeter_worst'),
  ('Area_worst', 'area_worst'),
  ('Smoothness_worst', 'smoothness_worst'),
  ('Compactness_worst', 'compactness_worst'),
  ('Concavity_worst', 'concavity_worst'),
  ('Concave points_worst', 'concave points_worst'),
  ('Symmetry_worst', 'symmetry_worst'),
  ('Fractal_dimension_worst', 'fractal_dimension_worst')]
 
    input_dict = {}

    for label,key in slider_labels:

       input_dict[key] =  st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()), 
            value = float(data[key].mean())
        )
    
    return input_dict

# Multi radar chart from Poltly
def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)
     
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
         r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=False
    )

    return fig

# Using model loading
def add_predictions(input_data):
   model = pickle.load(open('..\model\model.pkl', 'rb'))
   scaler = pickle.load(open('..\model\scaler.pkl', 'rb'))

   input_data_array = np.array(list(input_data.values())).reshape(1,-1)

   input_scaled_data = scaler.transform(input_data_array)

   prediction = model.predict(input_scaled_data)

   st.subheader("Cell cluster Prediction")
   st.write("The cell cluster is:")

   if prediction[0] == 'B': 
      st.write("<b>Benign</b>", unsafe_allow_html=True)
   else:
      st.write("<b>Malicious</b>", unsafe_allow_html=True)

   st.write("Probability of being Benign: ",round( model.predict_proba(input_scaled_data)[0][0],2))
   st.write("Probability of being Malicious: ",round( model.predict_proba(input_scaled_data)[0][1],2))

#main Entry of App
def main():
    st.set_page_config(
        page_title='Breast Cancer Predict', 
        layout="wide",
        initial_sidebar_state='expanded'
    )

    
    #sidebar
    inputs = add_sidebar()

    # structure of application
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

        col1, col2 = st.columns([4,2])

        with col1:
            radar = get_radar_chart(inputs)
            st.subheader("Graphical Representation")
            st.plotly_chart(radar)

        with col2:
           add_predictions(inputs)



if __name__ == '__main__':
    main()