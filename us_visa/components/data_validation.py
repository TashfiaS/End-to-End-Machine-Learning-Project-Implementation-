import sys

import pandas as pd
from scipy import stats

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise USvisaException(e, sys) from e

    def is_column_exist(self, df: pd.DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)
        except Exception as e:
            raise USvisaException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def detect_dataset_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        """
        KS-test for numerical columns, chi-square for categorical.
        Returns True if dataset-level drift is detected (>50% of features drifted).
        """
        try:
            numerical_cols = self._schema_config.get("numerical_columns", [])
            categorical_cols = self._schema_config.get("categorical_columns", [])

            drift_results = {}

            for col in numerical_cols:
                if col in reference_df.columns and col in current_df.columns:
                    stat, p_value = stats.ks_2samp(
                        reference_df[col].dropna(), current_df[col].dropna()
                    )
                    drift_results[col] = {"p_value": p_value, "drifted": p_value < threshold}

            for col in categorical_cols:
                if col in reference_df.columns and col in current_df.columns:
                    ref_counts = reference_df[col].value_counts()
                    cur_counts = current_df[col].value_counts()
                    all_cats = set(ref_counts.index) | set(cur_counts.index)
                    ref_freq = [ref_counts.get(c, 0) for c in all_cats]
                    cur_freq = [cur_counts.get(c, 0) for c in all_cats]
                    if sum(ref_freq) > 0 and sum(cur_freq) > 0:
                        stat, p_value = stats.chisquare(cur_freq, f_exp=[
                            r * sum(cur_freq) / sum(ref_freq) for r in ref_freq
                        ])
                        drift_results[col] = {"p_value": p_value, "drifted": p_value < threshold}

            n_features = len(drift_results)
            n_drifted = sum(1 for v in drift_results.values() if v["drifted"])
            dataset_drift = n_drifted > (n_features / 2)

            report = {
                "drift_detected": dataset_drift,
                "n_features": n_features,
                "n_drifted_features": n_drifted,
                "feature_drift": drift_results,
            }

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report,
            )

            logging.info(f"{n_drifted}/{n_features} features drifted. Dataset drift: {dataset_drift}")
            return dataset_drift
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            train_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.trained_file_path
            )
            test_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            # Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "

            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in test dataframe. "

            # Validate column existence
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += "Columns are missing in test dataframe. "

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info("Drift detected.")
                    validation_error_msg = "Drift detected."
                else:
                    validation_error_msg = "Drift not detected."
            else:
                logging.info(f"Validation error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
