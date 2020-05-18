#!/usr/bin/env python3

from flask import Flask, jsonify, request
import json
import pickle
import numpy as np
from sklearn.externals import joblib


app = Flask(__name__)


@app.route('/predict_region', methods = ['GET', 'POST'])
def predict_region():
    if request.method == 'POST':
            # Get the values of keys
            post_data = request.form
            age = post_data['age']
            degree = post_data['degree-of-diffe']
            small_intestine = post_data['small-intestine']
            sex = post_data['sex_2']
            histo_type_2 = post_data['histologic-type']
            bone = post_data['bone_2']
            bone_marrow = post_data['bone-marrow_2']
            lung = post_data['lung_2']
            pleura = post_data['pleura_2']
            peritoneum = post_data['peritoneum_2']
            liver = post_data['liver_2']
            brain = post_data['brain_2']
            skin = post_data['skin_2']
            neck = post_data['neck_2']
            supraclavicular = post_data['supraclavicular_2']
            axillar = post_data['axillar_2']
            mediastinum = post_data['mediastinum_2']
            abdominal = post_data['abdominal_2']

            # ['age', 'sex', 'histologic-type', 'degree-of-diffe', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum', 'liver',
             # 'brain', 'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal', 'small-intestine']

            new_data = np.array([[
                int(age), int(sex), int(histo_type_2), int(degree),  int(bone),  int(bone_marrow),  int(lung), int(pleura),
                int(peritoneum), int(liver), int(brain), int(skin), int(neck), int(supraclavicular), int(axillar),
                int(mediastinum), int(abdominal), int(small_intestine)
            ]])

            # Make Predictions using the pre-trained model
            model = joblib.load(open('resources/models/ParentModel.pkl', 'rb'))
            class_prediced = int(model.predict(new_data))
            return  jsonify(class_prediced)





@app.route('/predict_organ', methods=['GET', 'POST'])
def predict_organ():
    if request.method == 'POST':
        # Get the values of keys
        post_data = request.form
        region = post_data['region']
        age = post_data['age']
        degree = post_data['degree-of-diffe']
        small_intestine = post_data['small-intestine']
        sex = post_data['sex_2']
        histo_type_2 = post_data['histologic-type']
        bone = post_data['bone_2']
        bone_marrow = post_data['bone-marrow_2']
        lung = post_data['lung_2']
        pleura = post_data['pleura_2']
        peritoneum = post_data['peritoneum_2']
        liver = post_data['liver_2']
        brain = post_data['brain_2']
        skin = post_data['skin_2']
        neck = post_data['neck_2']
        supraclavicular = post_data['supraclavicular_2']
        axillar = post_data['axillar_2']
        mediastinum = post_data['mediastinum_2']
        abdominal = post_data['abdominal_2']

        # Columns[
        #     'age', 'sex', 'histologic-type', 'degree-of-diffe', 'bone', 'bone-marrow', 'lung', 'pleura', 'peritoneum',
        #     'liver', 'brain', 'skin', 'neck', 'supraclavicular', 'axillar', 'mediastinum', 'abdominal', 'small-intestine']

        # New instance
        new_data = np.array([[
            int(age), int(sex), int(histo_type_2), int(degree), int(bone), int(bone_marrow), int(lung), int(pleura),
            int(peritoneum), int(liver), int(brain), int(skin), int(neck), int(supraclavicular), int(axillar),
            int(mediastinum), int(abdominal), int(small_intestine)
        ]])


        # Make Predictions using the pre-trained model
        if(region == "1"):
            model = joblib.load(open('resources/models/URModel.pkl', 'rb'))
        elif(region == "2"):
            model = joblib.load(open('resources/models/TRModel.pkl', 'rb'))
        elif(region == "3"):
            model = joblib.load(open('resources/models/IPRModel.pkl', 'rb'))
        elif (region == "4"):
            model = joblib.load(open('resources/models/EPRModel.pkl', 'rb'))

        class_prediced = int(model.predict(new_data))
        return jsonify(class_prediced)






if __name__ == '__main__':
    app.run(debug=True)
    
