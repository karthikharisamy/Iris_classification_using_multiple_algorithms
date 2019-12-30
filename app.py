from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/load')
def index():
    return render_template("indexs.html")
     


@app.route('/predict',methods=["POST"])
def analyze():
	if request.method == 'POST':
		Sepal_Length = request.form['Sepal_Length']
		Sepal_Width = request.form['Sepal_Width']
		Petal_Length = request.form['Petal_Length']
		Petal_Width = request.form['Petal_Width']
		Model_choice = request.form['Model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [Sepal_Length,Sepal_Width,Petal_Length,Petal_Width]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if Model_choice == 'logitmodel':
		    logit_model = joblib.load('IrisModel_lr.pckl')
		    result_prediction = logit_model.predict(ex1)
		elif Model_choice == 'knnmodel':
			knn_model = joblib.load('IrisModel_knn.pckl')
			result_prediction = knn_model.predict(ex1)
		elif Model_choice == 'svmmodel':
			knn_model = joblib.load('IrisModel_svm.pckl')
			result_prediction = knn_model.predict(ex1)

	return render_template('indexs.html', Sepal_Length=Sepal_Length,
		Sepal_Width=Sepal_Width,
		Petal_Length=Petal_Length,
		Petal_Width=Petal_Width,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=Model_choice)


if __name__ == '__main__':
	app.run(debug=True)