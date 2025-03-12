from flask import Flask,request,jsonify,url_for,render_template 
import numpy as np
import pandas as pd
import pickle 

app=Flask(__name__)
pic_model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(i) for i in request.form.values()]
    data=np.array(data).reshape(1,-1)
    output=pic_model.predict(data)
    print(output[0])
    return render_template('home.html',predicted_value='{}'.format(output[0]))



if((__name__)=='__main__'):
    app.run(debug=True)
