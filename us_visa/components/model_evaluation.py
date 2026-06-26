import sys
import pandas as pd
import numpy as np

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
)
from us_visa.utils.main_utils import load_object, get_classification_score
from us_visa.ml.model.estimator import TargetValueMapping
from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.components.data_transformation import BINARY_YN_COLUMNS


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def get_best_model(self):
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            s3_client = SimpleStorageService()

            model_in_s3 = s3_client.s3_key_path_available(
                bucket_name=bucket_name, s3_key=model_path
            )
            if model_in_s3:
                model = s3_client.load_model(
                    model_name=model_path, bucket_name=bucket_name
                )
                return model
            return None
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _prepare_test_df(self) -> pd.DataFrame:
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df["company_age"] = CURRENT_YEAR - test_df["yr_of_estab"]
            test_df.drop(columns=["case_id", "yr_of_estab"], inplace=True, errors="ignore")
            for col in BINARY_YN_COLUMNS:
                test_df[col] = test_df[col].map({"Y": 1, "N": 0})
            return test_df
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            test_df = self._prepare_test_df()
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())

            # Score the newly trained model
            trained_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )
            trained_model_f1 = get_classification_score(
                y_true=y, y_pred=trained_model.predict(x)
            ).f1_score

            # Check for existing production model on S3
            best_model_in_s3 = self.get_best_model()
            is_model_accepted = True
            changed_accuracy = 0.0

            if best_model_in_s3 is not None:
                s3_model_f1 = get_classification_score(
                    y_true=y, y_pred=best_model_in_s3.predict(x)
                ).f1_score
                changed_accuracy = trained_model_f1 - s3_model_f1
                logging.info(
                    f"S3 model F1: {s3_model_f1:.4f} | New model F1: {trained_model_f1:.4f} | "
                    f"Delta: {changed_accuracy:.4f}"
                )
                if changed_accuracy < self.model_eval_config.changed_threshold_score:
                    is_model_accepted = False
                    logging.info("Trained model is not better than S3 model — rejecting.")
            else:
                logging.info("No model in S3. Accepting trained model.")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                changed_accuracy=changed_accuracy,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
