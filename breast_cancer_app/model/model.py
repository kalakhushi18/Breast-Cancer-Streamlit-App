import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def train_model(data):
   
   X = data.drop('diagnosis', axis = 1 )
   Y = data['diagnosis']

   #train-test split
   X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=42)

   #Scaling the data features
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   #training model
   model = LogisticRegression()

   model.fit(X_train_scaled, Y_train)

   Y_pred = model.predict(X_test_scaled)

   accuracy_result = accuracy_score(y_true=Y_test, y_pred=Y_pred)

   print(accuracy_result)

   classification_result = classification_report(y_true=Y_test, y_pred=Y_pred)

   print(classification_result)

   return model, scaler
 

#reading  data
def get_cleaned_data():
   data = pd.read_csv('..\data\data.csv')

   #IDA
   num_rows = data.shape[0]
   num_cols = data.shape[1]

   print("Number of Rows", num_rows)
   print("number of Cols", num_cols)
   
   #dropping duplicates
   data = data.drop_duplicates()

   #dropping unnecessary col
   data = data.drop(columns=['Unnamed: 32', 'id'])

    #exploratory data
   cols_list = data.columns
   data_description = data.describe().T
   
   return data, cols_list, data_description


def main():
   data_df,cols_list, data_description  = get_cleaned_data()
   
   print("columns list", cols_list)

   print("Data Description", data_description)

   trained_model, scaler = train_model(data = data_df)

   #creating model
   with open("model.pkl", mode="wb") as f:
      pickle.dump(trained_model, f)
      
   with open("scaler.pkl",mode="wb" ) as f:
      pickle.dump(scaler, f)



if __name__ == '__main__':
   main()

