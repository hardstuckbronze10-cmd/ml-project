import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # The preprocessor handles the None values by filling them with medians
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        sector: str,
        area: float,
        bedRoom: int,
        bathroom: int,
        balcony: str,
        floorNum: float,
        agePossession: str,
        has_servant_room: int,
        has_study_room: int,
        has_pooja_room: int,
        has_store_room: int,
        furnishing_type: int,
        built_up_area: float = None,
        carpet_area: float = None
    ):
        # Frontend fields only
        self.sector = sector
        self.area = area
        self.bedRoom = bedRoom
        self.bathroom = bathroom
        self.balcony = balcony
        self.floorNum = floorNum
        self.agePossession = agePossession
        self.has_servant_room = has_servant_room
        self.has_study_room = has_study_room
        self.has_pooja_room = has_pooja_room
        self.has_store_room = has_store_room
        self.furnishing_type = furnishing_type
        self.built_up_area = built_up_area
        self.carpet_area = carpet_area

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "sector": [self.sector],
                "area": [self.area],
                "bedRoom": [self.bedRoom],
                "bathroom": [self.bathroom],
                "balcony": [self.balcony],
                "floorNum": [self.floorNum],
                "agePossession": [self.agePossession],
                "has_servant_room": [self.has_servant_room],
                "has_study_room": [self.has_study_room],
                "has_pooja_room": [self.has_pooja_room],
                "has_store_room": [self.has_store_room],
                "furnishing_type": [self.furnishing_type],
                "built_up_area": [self.built_up_area],
                "carpet_area": [self.carpet_area],
                
                # Hidden from frontend - processed in background
                "has_others": [0],              # Fixed at 0
                "luxury_score": [None],         # Replaced with median
                "super_built_up_area": [None],  # Replaced with median
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)