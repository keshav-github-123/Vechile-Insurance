import sys
import pandas as pd
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging


class VehicleData:
    def __init__(
        self,
        Gender,
        Age,
        Driving_License,
        Region_Code,
        Previously_Insured,
        Annual_Premium,
        Policy_Sales_Channel,
        Vintage,
        Vehicle_Age,
        Vehicle_Damage
    ):
        try:
            self.Gender = Gender
            self.Age = float(Age)
            self.Driving_License = int(Driving_License)
            self.Region_Code = float(Region_Code)
            self.Previously_Insured = int(Previously_Insured)
            self.Annual_Premium = float(Annual_Premium)
            self.Policy_Sales_Channel = float(Policy_Sales_Channel)
            self.Vintage = int(Vintage)
            self.Vehicle_Age = Vehicle_Age
            self.Vehicle_Damage = Vehicle_Damage

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self) -> pd.DataFrame:
        try:
            vehicle_input_dict = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage],
            }
            return pd.DataFrame(vehicle_input_dict)
        except Exception as e:
            raise MyException(e, sys) from e


class VehicleDataClassifier:
    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig()):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config

            # Initialize S3 model loader + preprocessing
            self.model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
                preprocessor_path=self.prediction_pipeline_config.preprocessor_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe: pd.DataFrame):
        """
        Takes raw DataFrame, preprocessing + prediction is handled inside Proj1Estimator
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier")
            result = self.model.predict(dataframe)
            logging.info("Prediction successful")
            return result
        except Exception as e:
            raise MyException(e, sys) from e
