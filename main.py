from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()


origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    Feedback : str

diabetes_model = pickle.load(open('model_code.sav','rb'))

@app.post('/skillsync_prediction')
def skillsync_pred(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    Feedback = input_dictionary['Feedback']

    input_list=[Feedback]

    prediction = model_code.predict(input_list)

    print (prediction)

