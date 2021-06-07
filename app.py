import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import model

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    # '''
    # For rendering results on HTML GUI
    # '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    #
    # output = round(prediction[0], 2)
    #
    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    if request.method == "POST":
        f = request.files['recording']
        f.save('recording.wav')
        result  = model.predict('recording.wav')
            # result = 5.6
        print('file uploaded successfully\n')

        # return render_template('index.html', request="POST")
        return str(result)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)