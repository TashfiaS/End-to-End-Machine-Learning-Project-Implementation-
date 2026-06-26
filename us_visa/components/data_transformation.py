import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN

from us_visa.constants import (
    TARGET_COLUMN, CURRENT_YEAR, SCHEMA_FILE_PATH
)
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
)
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import (
    save_object, save_numpy_array_data, read_yaml_file, drop_columns
)
from us_visa.ml.model.estimator import TargetValueMapping


# Column groups for the preprocessing pipeline
NUMERICAL_COLUMNS = ["no_of_employees", "company_age", "prevailing_wage"]

OR_COLUMN = "education_of_employee"
OR_CATEGORIES = [["High School", "Associate", "Bachelor's", "Master's", "Doctorate"]]

OH_COLUMNS = ["continent", "region_of_employment", "unit_of_wage"]

# Binary Y/N columns mapped manually before pipeline
BINARY_YN_COLUMNS = ["has_job_experience", "requires_job_training", "full_time_position"]


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entered get_data_transformer_object of DataTransformation class")
        try:
            ordinal_pipeline = Pipeline(
                steps=[("ordinal_encoder", OrdinalEncoder(categories=OR_CATEGORIES))]
            )

            onehot_pipeline = Pipeline(
                steps=[
                    ("onehot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            numerical_pipeline = Pipeline(
                steps=[("scaler", StandardScaler())]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, NUMERICAL_COLUMNS),
                    ("ordinal_pipeline", ordinal_pipeline, [OR_COLUMN]),
                    ("onehot_pipeline", onehot_pipeline, OH_COLUMNS),
                    # Binary columns are already 0/1 after manual mapping
                    ("binary_passthrough", "passthrough", BINARY_YN_COLUMNS),
                ]
            )

            logging.info("Created preprocessor object")
            return preprocessor
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")

                preprocessor = self.get_data_transformer_object()

                train_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.trained_file_path
                )
                test_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                )

                # Drop schema-defined drop columns (case_id)
                drop_cols = self._schema_config.get("drop_columns", [])

                # Feature engineering: company_age from yr_of_estab
                train_df["company_age"] = CURRENT_YEAR - train_df["yr_of_estab"]
                test_df["company_age"] = CURRENT_YEAR - test_df["yr_of_estab"]

                train_df = drop_columns(train_df, cols=drop_cols + ["yr_of_estab"])
                test_df = drop_columns(test_df, cols=drop_cols + ["yr_of_estab"])

                # Map binary Y/N columns to 0/1
                for col in BINARY_YN_COLUMNS:
                    train_df[col] = train_df[col].map({"Y": 1, "N": 0})
                    test_df[col] = test_df[col].map({"Y": 1, "N": 0})

                # Separate features and target
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info("Applying target mapping")
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )
                target_feature_test_df = target_feature_test_df.replace(
                    TargetValueMapping()._asdict()
                )

                # Fit preprocessor on train, transform both
                logging.info("Fitting and transforming using preprocessor")
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                # Apply SMOTEENN to handle class imbalance on training data
                logging.info("Applying SMOTEENN for imbalanced dataset")
                smt = SMOTEENN(sampling_strategy="minority")
                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )

                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

                # Save transformed arrays and preprocessor
                save_numpy_array_data(
                    self.data_transformation_config.transformed_train_file_path, array=train_arr
                )
                save_numpy_array_data(
                    self.data_transformation_config.transformed_test_file_path, array=test_arr
                )
                save_object(
                    self.data_transformation_config.transformed_object_file_path,
                    preprocessor,
                )
                logging.info("Saved preprocessor and transformed arrays")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                )
                logging.info(f"Data transformation artifact: {data_transformation_artifact}")
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
        except Exception as e:
            raise USvisaException(e, sys) from e
