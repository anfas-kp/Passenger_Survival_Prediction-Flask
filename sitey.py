from flask import Flask,request,render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


app=Flask(__name__)

model=pickle.load(open('C:/Users/Pc/OneDrive/Desktop/USING_FLASK/survived_modell.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',prediction_text='')

@app.route('/predict',methods=['POST'])
def predict():
    Passenger_ID=(request.form['Passenger_ID'])
    Age=float(request.form['Age'])
    Gender=str(request.form['Gender'])
    Class=str(request.form['Class'])
    Seat_Type=str(request.form['Seat_Type'])
    Fare_Paid=float(request.form['Fare_Paid'])

    
    input_data = np.array([[Passenger_ID,Age,Gender,Class,Seat_Type,Fare_Paid]])

    new_data=pd.DataFrame(input_data,columns=['Passenger_ID','Age','Gender','Class','Seat_Type','Fare_Paid'])

    new_data['Gender']=new_data['Gender'].map({'Male':1,'Female':2})
    new_data['Class']=new_data['Class'].map({'Economy':1,'First':2,'Business':3})
    new_data['Seat_Type']=new_data['Seat_Type'].map({'Window':1,'Middle':2,'Aisle':3})
    
    prediction = model.predict(new_data)

    
    result='YOU SURVIVED' if prediction[0]==1 else 'YOU DEAD'
    prediction_text=f'{result}'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)