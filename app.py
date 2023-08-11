import numpy as np
import pandas as pd
import pickle
from flask import Flask , request , url_for , render_template , app , jsonify

def load_model(path , way = 1):
    if way == 1:
        total = pickle.load(open(path, "rb"))
    ## Or way two:
    elif way == 2:
        with open(path, "rb") as f:
            total = pickle.load(f)
    
    model = total["model"]

    return model

# print(model1 , model)
path = "model.pkl" 
model= load_model(path  , 1)


app = Flask(__name__)

@app.route("/", methods = ['GET' , 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict" , methods = ['POST'])
def predict():
    bedrooms = request.form['PN']
    restrooms = request.form['WC']
    floors = request.form['Floors']
    provinces = request.form['Provinces']
    tien_ich = request.form['Tien_ich']
    area = request.form['Areas']
    quan_huyen = request.form['Quan/Huyen']

    input_values = pd.DataFrame([[provinces , quan_huyen , bedrooms , restrooms , area , floors , tien_ich]] , 
                                columns= ["Provinces"	,"Quan/Huyen"	,"PN"	,"WC",	"Areas",	"Floors"	,"Tien_ich"])

    output = model.predict(input_values)[0]
    print(output)
    return render_template('predict.html' , prediction_text = f"~{output:.1f}")







if __name__ == "__main__":
    app.run(debug= True)
# print(model)
