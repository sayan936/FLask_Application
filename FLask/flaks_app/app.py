from flask import Flask,request,jsonify
from joblib import load
from flask_cors import CORS, cross_origin

model = load('./../ml_model/saved_models/model.joblib')
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/',methods = ['GET','POST'])
@cross_origin()
def main():
    if request.method == 'POST':
        data = request.get_json()
        value = [x for x in data.values()]
        y_hat = model.predict([value])
        
        return jsonify({'predicted_value':str(y_hat[0])}) 
           
    return ''

if __name__ == "__main__":
    app.run(debug = True)