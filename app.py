from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)



filename = 'Lung_Cancer.pkl'
with open(filename, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def hello_world():
    return render_template("lung_cancer.html" ,pred='Please fill all fields')



@app.route('/predict',methods=['POST', 'GET'])
def predict():
        if request.method == 'POST':
            int_features = [int(x) for x in request.form.values()]
            final = np.reshape(int_features, (1, -1))
            print(int_features)  #Checking Inputs Successfully Added
            print(final) #Reshaping into numpy array for Prediction
            prediction = model.predict(final)
            print(prediction) # Checking the Prediction Value
            output = prediction
            if output == 0:
                return render_template('lung_cancer.html', pred='You May Not Got Lung Cancer')
            else:
                return render_template('lung_cancer.html', pred='You May Have Lung Cancer ')
        else:
            return render_template('lung_cancer.html')

if __name__ == '__main__':
    app.run(debug=True)
