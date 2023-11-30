from flask import Flask
from flask_cors import CORS
from flask import request
from flask import render_template 

import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open("svc_model.pkl", "rb"))
        
@app.route("/")
def homePage():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template('index.html', prediction_details = "The Machine Learning model predicts that patient may have class {} cancer.".format(prediction))

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
