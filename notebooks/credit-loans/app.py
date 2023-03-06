import gradio as gr
import numpy as np
from PIL import Image
import hopsworks
import joblib
import os
from features import loans

key=""
with open("api-key.txt", "r") as f:
    key = f.read().rstrip()
os.environ['HOPSWORKS_PROJECT']="deloitte"
os.environ['HOPSWORKS_HOST']="6a525ee0-91d8-11ed-9cc8-9fe82dc2b6fd.cloud.hopsworks.ai"
os.environ['HOPSWORKS_API_KEY']=key

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("lending_model", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/lending_model.pkl")

fv = fs.get_feature_view("loans", version=1)

#purpose = ['vacation', 'debt_consolidation', 'credit_card','home_improvement', 'small_business', 'major_purchase', 'other',
#       'medical', 'wedding', 'car', 'moving', 'house', 'educational','renewable_energy']
#term = [' 36 months', ' 60 months']


fv.init_serving(training_dataset_version=4)

def approve_loan(id, term, purpose, loan_amnt, int_rate):
    input_list = []
    #input_list.append(f2)
    encoded_term = term
    encoded_purpose = purpose
    encoded_loan_amnt = loan_amnt
    encoded_int_rate = int_rate
     
    input_features = fv.get_feature_vector({"id": id}, passed_features={"term": encoded_term, "purpose": encoded_purpose, "loan_amnt": encoded_loan_amnt, "int_rate": encoded_int_rate})

    y_pred = model.predict(input_features)
    #res = model.predict(np.asarray(input_features).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/setosa.png"
    if y_pred == 1:
        flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/virginica.png"
    img = Image.open(requests.get(flower_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=approve_loan,
    title="Loan Approval",
    description="Enter your details to see if your loan will be approved or not.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(label="id"),
        gr.inputs.Dropdown(label="term", options=term),
        gr.inputs.Dropdown(label="purpose", options=purpose),
        gr.inputs.Number(default=1000, label="loan_amnt"),
        gr.inputs.Number(default=4.0, label="int_rate"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()
