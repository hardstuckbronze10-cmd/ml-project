from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# List of Gurgaon sectors to populate the dropdown dynamically.
# Update these values to match exactly what your model expects!
SECTORS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', 
    '15', '17', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 
    '31', '33', '37C', '37D', '43', '45', '47', '48', '49', '50', '52', '53', 
    '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '65', '66', 
    '67', '68', '69', '70', '71', '72', '73', '74', '76', '77', '78', '79', 
    '80', '81', '82', '83', '84', '85', '86', '88', '89', '90', '91', '92', 
    '93', '95', '99', '102', '103', '104', '105', '106', '107', '108', '109', 
    '110', '111', '112', '113', 'Sohna', 'Dwarka Expressway'
]

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Pass the sectors list to the template
        return render_template('home.html', sectors=SECTORS)
    else:
        data = CustomData(
            sector=request.form.get('sector'),
            area=float(request.form.get('area')),
            bedRoom=int(request.form.get('bedRoom')),
            bathroom=int(request.form.get('bathroom')),
            balcony=request.form.get('balcony'),
            floorNum=float(request.form.get('floorNum')),
            agePossession=request.form.get('agePossession'),
            has_servant_room = 1 if request.form.get('has_servant_room') else 0,
            has_study_room = 1 if request.form.get('has_study_room') else 0,
            has_pooja_room = 1 if request.form.get('has_pooja_room') else 0,
            has_store_room = 1 if request.form.get('has_store_room') else 0,
            furnishing_type=int(request.form.get('furnishing_type')),
            built_up_area=float(request.form.get('built_up_area')) if request.form.get('built_up_area') else None,
            carpet_area=float(request.form.get('carpet_area')) if request.form.get('carpet_area') else None
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        print("Before Prediction")
        predict_pipeline = PredictPipeline()
        
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        
        print("After Prediction")
        
        # Pass the sectors list here too so the dropdown still works after a prediction
        return render_template('home.html', results=round(results[0], 2), sectors=SECTORS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)