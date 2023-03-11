import gradio as gr
import numpy as np
from PIL import Image
import hopsworks
import joblib
import os
import pandas as pd
from features import loans
import requests

key=""
with open("api-key.txt", "r") as f:
    key = f.read().rstrip()
os.environ['HOPSWORKS_PROJECT']="deloitte"
os.environ['HOPSWORKS_HOST']="6a525ee0-91d8-11ed-9cc8-9fe82dc2b6fd.cloud.hopsworks.ai"
os.environ['HOPSWORKS_API_KEY']=key

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("lending_model", version=7)
model_dir = model.download()
model = joblib.load(model_dir + "/lending_model.pkl")

fv = fs.get_feature_view("loans", version=1)

purpose = ['vacation', 'debt_consolidation', 'credit_card','home_improvement', 'small_business', 'major_purchase', 'other',
       'medical', 'wedding', 'car', 'moving', 'house', 'educational','renewable_energy']
term = [' 36 months', ' 60 months']


fv.init_serving(training_dataset_version=4)
print("Initialized feature view for serving")

def approve_loan(id, term, purpose, zip_code, loan_amnt, int_rate):
    input_list = []
    #input_list.append(f2)
    encoded_term = term
    encoded_purpose = purpose
    encoded_loan_amnt = loan_amnt
    encoded_int_rate = int_rate
    
    # On-demand feature function used to create the zip_code feature
    validated_zip_code = loans.zipcode(zip_code)
    if validated_zip_code == 0:
        raise Exception('Invalid zip code. It should have 5 digits')
        
        
    print("Requesting Feature Vector")
    arr = fv.get_feature_vector({"id": id})
    print("Received Feature Vector: {}".format(arr))
    dict = {'earliest_cr_line_year': arr[0], 'loan_amnt': arr[1], 'term': arr[2], 'int_rate': arr[3],
           'installment':arr[4], 'sub_grade':arr[5], 'home_ownership':arr[6], 'annual_inc':arr[7], 
            'verification_status': arr[8], 'purpose': arr[9], 'dti':arr[10], 'open_acc':arr[11], 
           'pub_rec':arr[12], 'revol_bal':arr[13], 'revol_util':arr[14], 'total_acc':arr[15], 
            'initial_list_status' : arr[16], 'application_type': arr[17],
           'mort_acc':arr[18], 'pub_rec_bankruptcies':arr[19], 'zip_code' : arr[20]} 

    print(dict)    
    arr = fv.get_feature_vector({"id": id}, passed_features={"term": encoded_term, "purpose": encoded_purpose,
                                                             "zip_code": validated_zip_code,
                                                             "loan_amnt": encoded_loan_amnt, 
                                                             "int_rate": encoded_int_rate})
    print("Received Feature Vector: {}".format(arr))

    dict = {'earliest_cr_line_year': arr[0], 'loan_amnt': arr[1], 'term': arr[2], 'int_rate': arr[3],
           'installment':arr[4], 'sub_grade':arr[5], 'home_ownership':arr[6], 'annual_inc':arr[7], 
            'verification_status': arr[8], 'purpose': arr[9], 'dti':arr[10], 'open_acc':arr[11], 
           'pub_rec':arr[12], 'revol_bal':arr[13], 'revol_util':arr[14], 'total_acc':arr[15], 
            'initial_list_status' : arr[16], 'application_type': arr[17],
           'mort_acc':arr[18], 'pub_rec_bankruptcies':arr[19], 'zip_code' : arr[20]} 

    print(dict)

    
    df = pd.DataFrame([dict])
    y_pred = model.predict(df)
    print("Prediction: {}".format(y_pred))
    #res = model.predict(np.asarray(input_features).reshape(1, -1)) 
    #np.asarray(feature_vector).reshape(1, -1)
    
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    loan_res_url = "https://icl-blog.s3.ap-southeast-1.amazonaws.com/uploads/2015/01/loan_approved.jpg"
    if y_pred == 0:
        loan_res_url = "https://elevatecredit.africa/wp-content/uploads/2022/03/download-2.jpg"
    img = Image.open(requests.get(loan_res_url, stream=True).raw)            
    return img

demo = gr.Interface(
    fn=approve_loan,
    title="Loan Approval",
    description="Enter your details to see if your loan will be approved or not.",
    allow_flagging="never",
    inputs=[
        gr.Number(label="id"),
        gr.Dropdown(term, label="term"),
        gr.Dropdown(purpose, label="purpose"),
        gr.Number(label="zip_code"),
        gr.Number(label="loan_amnt"),
        gr.Number(label="int_rate"),
        ],
    examples=[
        [2222, "36 months","home_improvement", 45725, 5000, 4.5],
    ],
    outputs=gr.Image(type="pil"))

demo.launch()

