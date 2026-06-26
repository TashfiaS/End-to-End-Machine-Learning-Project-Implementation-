import os
import shutil
import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.constants import SAVED_MODEL_PATH

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.model_trainer import ModelTrainer

from us_visa.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            return artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed")
            return artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed")
            return artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training completed")
            return artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

    def save_model_locally(self, trained_model_path: str) -> None:
        """Copy the trained model to a fixed local path for prediction."""
        try:
            os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
            shutil.copy(trained_model_path, SAVED_MODEL_PATH)
            logging.info(f"Model saved locally at: {SAVED_MODEL_PATH}")
        except Exception as e:
            raise USvisaException(e, sys) from e

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact, data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            self.save_model_locally(model_trainer_artifact.trained_model_file_path)
            logging.info(
                f"Pipeline complete. Model: {SAVED_MODEL_PATH} | "
                f"F1: {model_trainer_artifact.metric_artifact.f1_score:.4f}"
            )
        except Exception as e:
            raise USvisaException(e, sys) from e
