from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame
import pandas as pd


class Proj1Estimator:
    """
    This class is used to save and retrieve our model from s3 bucket and to do prediction
    """

    def __init__(self,bucket_name,model_path,preprocessor_path=None):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.loaded_preprocessor = None
        self.loaded_model:MyModel=None


    def is_model_present(self,model_path):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except MyException as e:
            print(e)
            return False

    def load_model(self,)->MyModel:
        """
        Load the model from the model_path
        :return:
        """

        return self.s3.load_model(self.model_path,bucket_name=self.bucket_name)
    

    def load_preprocessor(self):
        """
        Load the model from the model_path
        :return:
        """

        return self.s3.load_model(self.preprocessor_path,bucket_name=self.bucket_name)


    def save_model(self,from_file,remove:bool=False, ml_model:bool=False, preprocessor_model=False)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:

            if ml_model:
                self.s3.upload_file(from_file,
                                    to_filename=self.model_path,
                                    bucket_name=self.bucket_name,
                                    remove=remove
                                    )
            if preprocessor_model:
                self.s3.upload_file(from_file,
                                    to_filename=self.preprocessor_path,
                                    bucket_name=self.bucket_name,
                                    remove=remove
                                    )

        except Exception as e:
            raise MyException(e, sys)
        
    
    def preprocess(self,df:DataFrame):
        """
        :param dataframe:
        :return: dataframe
        """
        try:
            self.loaded_preprocessor = self.load_preprocessor()

            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)

            # Step 2: Apply same one-hot logic for Vehicle_Age and Vehicle_Damage
            df['Vehicle_Age_lt_1_Year'] = (df['Vehicle_Age'] == '< 1 Year').astype(int)
            df['Vehicle_Age_gt_2_Years'] = (df['Vehicle_Age'] == '> 2 Years').astype(int)
            df['Vehicle_Damage_Yes'] = (df['Vehicle_Damage'] == 'Yes').astype(int)

            # Drop original string columns
            df.drop(['Vehicle_Age', 'Vehicle_Damage'], axis=1, inplace=True)

            transformed = self.loaded_preprocessor.transform(df)

            # Step 4: Get feature names (cleaned like before)
            feature_names = self.loaded_preprocessor.named_steps['Preprocessor'].get_feature_names_out()
            feature_names = [col.replace('StandardScaler__', '')
                                .replace('MinMaxScaler__', '')
                                .replace('remainder__', '') for col in feature_names]

            # Step 5: Return as DataFrame
            df_transformed = pd.DataFrame(transformed, columns=feature_names)
            return df_transformed
        
        except Exception as e:
            raise MyException(e, sys)



    def predict(self,dataframe:DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            dataframe = self.preprocess(df=dataframe)
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise MyException(e, sys)