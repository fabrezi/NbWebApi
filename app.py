from flask import Flask, session, request
from flask_restful import Api
from NbRequest import NBPredict

app = Flask(__name__)
api = Api(app)
api.add_resource(NBPredict, "/predict")
app.run(debug=True)
