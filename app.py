import numpy as np
from flask import Flask, abort, jsonify, request, make_response
from flask_restx import Api, Resource, fields
import pickle
from titanic import create_titanic_pkl
import os.path
from os import path
import re
import sys


app = Flask(__name__)
api = Api(app, version='1.0', title='EXL',
    description='# Titanic prediction API',)
# app.config['SWAGGER_UI_JSONEDITOR'] = False

ns = api.namespace(name='API List | Click Here', path='/api/v1')

predicted_data = []

@ns.route('/clear', doc={"description": "# Clear the data for this session."},)
@ns.header('Create File', 'Generate Pickle File.')
class CreatePickle(Resource):
    def get(self):
        global predicted_data
        predicted_data = []
        return { "Session Data": "Cleared" }

@ns.route('/model', doc={"description": "Create the Model pickle file."},)
@ns.header('Create File', 'Generate Pickle File.')
class CreatePickle(Resource):
    def get(self):
        if not path.exists("titanic_rfc.pkl"):
            create_titanic_pkl()
            return {'File': 'Created'}
        else:
            return {'File': 'already exists'}
        
my_rfc = pickle.load(open('titanic_rfc.pkl', 'rb'))

# # pred = api.model('Predict', {'predict_this' : fields.String('dict of sl,sw,pl,pw')})
pred = api.model('Predict', {
                    'Pclass' : fields.Integer(required=True, description='# Ticket class - Integer | input must be 1 or 2 or 3', example=1, min=1, max=3), 
                    'Name' : fields.String(required=True, description='# Passenger Name with Title followed by "."', example = 'Braund, Mr. Owen Harris'),
                    'Sex' : fields.String(required=True, description='# Sex - male or female', enum=['male','female']),
                    'Age' : fields.Float(required=True, description='# Age in years', example=29.0),
                    'SibSp': fields.Integer(required=True, description='# Integer | Number of siblings / spouses aboard the Titanic', example=2, min=0),
                    'Parch': fields.Integer(required=True, description='# Integer | Number of parents / children aboard the Titanic', example=1, min=0),
                    'Ticket': fields.String(description='# Ticket number - not mandatory', example='PC 17599'),
                    'Fare': fields.Float(required=True, description='# Passenger Fare', example=32.02),
                    'Cabin': fields.String(required=True, description='# Cabin number', example="C123"),
                    'Embarked': fields.String(required=True, description='# Port of Embarkation', enum=['S', 'C', 'Q']),
                    })

def pred_func(data):
    # gender encoding
    genders = {"male": 0, "female": 1}
    data['Sex'] = genders[data['Sex']]

    # Age Binning
    data['Age'] = int(data['Age'])
    if data['Age'] <= 11:
        data['Age'] = 0
    elif data['Age'] <= 18:
        data['Age'] = 1
    elif data['Age'] <= 22:
        data['Age'] = 2
    elif data['Age'] <= 27:
        data['Age'] = 3
    elif data['Age'] <= 33:
        data['Age'] = 4
    elif data['Age'] <= 40:
        data['Age'] = 5
    else:
        data['Age'] = 6

    # Relatives Count and Not Alone 
    data['relatives'] = data['SibSp'] + data['Parch']
    if data['relatives'] > 0:
        data['not_alone'] = 0
    elif['relatives'] == 0:
        data['not_alone'] = 1
    else:
        data['not_alone'] = 0
        print('SibSp and Parch can not add to negative')

    # Fare to Binning
    if data['Fare'] <= 7.91:
        data['Fare'] = 0
    elif data['Fare'] <= 14.454:
        data['Fare'] = 1
    elif data['Fare'] <= 31:
        data['Fare'] = 2
    elif data['Fare'] <= 99:
        data['Fare'] = 3
    elif data['Fare'] <= 250:
        data['Fare'] = 4
    else:
        data['Fare'] = 5
    
    # Embarked encoding
    ports = {"S": 0, "C": 1, "Q": 2}
    data['Embarked'] = ports[data['Embarked']]

    # Cabin Number changed to Deck Type
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    if data['Cabin'][0] in deck.keys():
        data['Deck'] = deck[data['Cabin'][0]]
    else:
        data['Deck'] = deck['U']

    # Title from the Name
    titles = {"Mr": 1, "Miss": 2, "Mlle": 2, "Ms": 2, "Mrs": 3, 
                "Mme": 3, "Master": 4, "Rare": 5, }
    titlename = re.search('([A-Za-z]+)\.' , data['Name']).group()
    if titlename in titles.keys():
        data['Title'] = titles[titlename[:-1]]
    else:
        data['Title'] = titles['Rare']

    # New columns
    data['Age_Class'] = data['Age'] * data['Pclass']
    data['Fare_Per_Person'] = data['Fare']//(data['relatives']+1)

    # terminal print
    print([[data['Pclass'], data['Sex'], data['Age'], data['SibSp'],
                         data['Parch'], data['Fare'], data['Embarked'], 
                         data['relatives'], data['not_alone'], data['Deck'], 
                         data['Title'], data['Age_Class'], data['Fare_Per_Person']]])

    predict_request = [[data['Pclass'], data['Sex'], data['Age'], data['SibSp'],
                         data['Parch'], data['Fare'], data['Embarked'], 
                         data['relatives'], data['not_alone'], data['Deck'], 
                         data['Title'], data['Age_Class'], data['Fare_Per_Person']]]
    predict_request = np.array(predict_request)
    y_hat = my_rfc.predict(predict_request)
    output = y_hat[0]
    b = {0:'Survived', 1: 'Not Survived'}
    return b[output]

@ns.route("/sessiondata", doc={"description": "Get the data for current session."},)
@ns.header('Data', 'Current Session data.')
class GetDataClass(Resource):
    def get(self):
        global predicted_data
        if not predicted_data:
            return "No prediction made so far."
        return predicted_data

@ns.route("/one", doc={"description": "Please input single json for prediction."},)
@ns.header('Single Predictor', 'Predicts only one.')
class OnePredictClass(Resource):
    @api.expect(pred)
    def post(self):
        try:
            global predicted_data
            data1 = api.payload
            data = data1.copy()
            result = pred_func(data)
            data1['Predicted_survival_status'] = result
            predicted_data.append(data1)
            return { 'Name': data1['Name'], 
                    'Cabin': data1['Cabin'],
                    'Predicted_survival_status': data1['Predicted_survival_status']}
        except:
            error_name = sys.exc_info()[0].__name__
            error_desc = sys.exc_info()[1] 
            return { "Incorrect JSON" : str(error_name) + " | " + str(error_desc) }

@ns.route("/many", doc={"description": "Please input multiple json in list form for predictions."},)
@ns.header('Multiple Predictor', 'Predicts many outputs.')
class ManyPredictClass(Resource):
    @api.expect([pred])
    def post(self):
        # main_data = api.payload
        # data = main_data['predict_this']
        main_data = api.payload
        a = []
        count = 0
        for data1 in main_data:
            try:
                global predicted_data
                count+=1
                data = data1.copy()
                result = pred_func(data)
                data1['Predicted_survival_status'] = result
                predicted_data.append(data1)
                a.append({ 'Name': data1['Name'], 
                    'Cabin': data1['Cabin'],
                    'Predicted_survival_status': data1['Predicted_survival_status']})
            except:
                error_name = sys.exc_info()[0].__name__
                error_desc = sys.exc_info()[1] 
                a.append({ "Incorrect JSON " + str(count) : str(error_name) + " | " + str(error_desc) })
        return a

if __name__ == "__main__":
    app.run()