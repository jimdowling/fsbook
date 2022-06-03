import os
import joblib
import xgboost

class Predict(object):

    def __init__(self):
        """Prepare and load a trained model"""
        # NOTE: The env var ARTIFACT_FILES_PATH contains the path to the model artifact files
        
        # load the sklearn pipeline object
        self.sktransformer = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/minmax-transformer.pkl")
        # load the trained model
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/review.pkl")

    def predict(self, inputs):
        """Serve prediction using a trained model"""
        # transform inputs with minmax scalar transformer
        input = self.sktransformer.transform(inputs)
        # make prediction
        return self.model.predict(inputs).tolist()  # Numpy Arrays are note JSON serializable