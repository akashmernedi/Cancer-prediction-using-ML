from flask import Flask
from flask_cors import CORS
from flask import request
from flask import render_template 
from pymongo import MongoClient

import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open("svc_model.pkl", "rb"))

def get_db():
    client = MongoClient("mongodb://bW9uZ291c2Vy:bW9uZ29wYXNz@mongodb-service:27017/admin")
    db = client['cancer_db']
    return db

def insert_data_to_db(data):
    db = get_db()
    collection = db['patient_data']
    collection.insert_one(data)
    print("Data inserted successfully!")
        
@app.route("/")
def homePage():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    data_to_store = {'features': float_features, 'prediction': prediction[0]}
    insert_data_to_db(data_to_store)
    
    return render_template('index.html', prediction_details = "The Machine Learning model predicts that patient may have class {} cancer.".format(prediction))

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
