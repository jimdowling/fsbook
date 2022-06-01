import os
import joblib
import xgboost

class Predict(object):

    def __init__(self):
        """Prepare and load a trained model"""
        # NOTE: The env var ARTIFACT_FILES_PATH contains the path to the model artifact files
        
        # load the trained model
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/review.pkl")

    def predict(self, inputs):
        """Serve prediction using a trained model""" # Numpy Arrays are note JSON serializable
        return self.model.predict(inputs).tolist()