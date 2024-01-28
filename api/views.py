# calculator_api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializer import SpeachSerializer
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import joblib
from .mymodels import LanguageDetectionModel , UrduHateSpeechModel , EnglishModel

current_directory = os.path.dirname(__file__)
label_encoder_path = os.path.join(current_directory, 'Language_label_encoder.pkl')
label_encoder = joblib.load(label_encoder_path)
vectorizer_path = os.path.join(current_directory, 'Language_vectorizer.pkl')
vectorizer = joblib.load(vectorizer_path)

class HateSpeachFinder(APIView):
    def __init__(self):
        super(HateSpeachFinder, self).__init__()

        
        self.model = self.load_model()

    def load_model(self):
        
        current_directory = os.path.dirname(__file__)
        model_path = os.path.join(current_directory, 'LanguageDetector.pth')

        if os.path.exists(model_path):
            input_size = 15570
            hidden_size = 128
            num_classes = 2
            model = LanguageDetectionModel(input_size , hidden_size , num_classes)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
    def EnglishDetector(self , text):
        
        english_model = EnglishModel(14413 , 128 , 2 , 2)
        current_directory = os.path.dirname(__file__)
        english_model_path = os.path.join(current_directory, 'English_Language_Model.pth')
        english_model.load_state_dict(torch.load(english_model_path)) 

        #load English vectorizer and labelencoder
        english_vectorizer_path = os.path.join(current_directory , "English_Vectorizer.pkl")
        english_vectorizer = joblib.load(english_vectorizer_path)

        english_label_encoder_path = os.path.join(current_directory , "English_Label_Encoder.pkl")
        english_label_encoder = joblib.load(english_label_encoder_path)

        # Perform count vectorization on user input
        user_input_vec = english_vectorizer.transform([text])
        user_input_tensor = torch.tensor(user_input_vec.toarray(), dtype=torch.float32)

        # Get the model's prediction
        with torch.no_grad():
            prediction = english_model(user_input_tensor)
            _, predicted_label = torch.max(prediction, 1)

        predicted_class = english_label_encoder.classes_[predicted_label.item()]

        return predicted_class
    
    def UrduDetector(self , text):

        urdumodel = UrduHateSpeechModel(8441 , 128 , 2 )  
        current_directory = os.path.dirname(__file__)
        urdu_model_path = os.path.join(current_directory, 'UrduLanguageModel.pth') 
        urdumodel.load_state_dict(torch.load(urdu_model_path))

        #load urdu vectorizer and labelencoder
        urdu_vectorizer_path = os.path.join(current_directory , "UrduModelVectorizer.pkl")
        urdu_vectorizer = joblib.load(urdu_vectorizer_path)

        urdu_label_encoder_path = os.path.join(current_directory , "Urdu_Lable_Encoder.pkl")
        urdu_label_encoder = joblib.load(urdu_label_encoder_path)

        # Perform count vectorization on user input
        user_input_vec = urdu_vectorizer.transform([text])
        user_input_tensor = torch.tensor(user_input_vec.toarray(), dtype=torch.float32)

        # Get the model's prediction
        with torch.no_grad():
            prediction = urdumodel(user_input_tensor)
            _, predicted_label = torch.max(prediction, 1)

        predicted_class = urdu_label_encoder.classes_[predicted_label.item()]

        return predicted_class



    def post(self, request, format=None):
        serializer = SpeachSerializer(data=request.data)

        if serializer.is_valid():
            data = serializer.validated_data
            text = data["text"]
            auth = data["auth"]

            if auth == "akashmemon@2001731":

                # Perform count vectorization on user input
                user_input_vec = vectorizer.transform([text])
                user_input_tensor = torch.tensor(user_input_vec.toarray(), dtype=torch.float32)
                
                

                # Get the model's prediction
                with torch.no_grad():
                    prediction = self.model(user_input_tensor)
                    _, predicted_label = torch.max(prediction, 1)

                predicted_class = label_encoder.classes_[predicted_label.item()]

                label = ""

                if predicted_class == "English" :
                    label = self.EnglishDetector(text)

                else:
                    label = self.UrduDetector(text)


                result = {
                    "Language_detected": predicted_class ,
                    "label" : label
                }

                return Response(result, status=status.HTTP_200_OK)
            else:
                result = {
                    "text": "Invalid Authentication Key" 
                    
                }
                return Response(result, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


   
