import sys

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import (
    load_numpy_array_data, save_object, load_object, get_classification_score
)
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from us_visa.ml.model.estimator import USvisaModel

MODELS = {
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [6, 9],
            "criterion": ["gini", "entropy"],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(eval_metric="logloss", random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
    },
}


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise USvisaException(e, sys)

    def train_model(self, X_train, y_train):
        try:
            best_model = None
            best_score = 0.0

            for name, config in MODELS.items():
                logging.info(f"Grid searching: {name}")
                gs = GridSearchCV(
                    estimator=config["estimator"],
                    param_grid=config["params"],
                    cv=3,
                    scoring="f1",
                    n_jobs=-1,
                    verbose=1,
                )
                gs.fit(X_train, y_train)
                logging.info(f"{name} best CV F1: {gs.best_score_:.4f} | params: {gs.best_params_}")

                if gs.best_score_ > best_score:
                    best_score = gs.best_score_
                    best_model = gs.best_estimator_

            logging.info(f"Best model: {best_model.__class__.__name__} | CV F1: {best_score:.4f}")
            return best_model
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            best_model = self.train_model(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_train, y_train_pred)

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_test, y_test_pred)

            logging.info(f"Train F1: {classification_train_metric.f1_score:.4f} | "
                         f"Test F1: {classification_test_metric.f1_score:.4f}")

            if classification_test_metric.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Model test F1 {classification_test_metric.f1_score:.4f} is below "
                    f"expected {self.model_trainer_config.expected_accuracy}"
                )

            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            usvisa_model = USvisaModel(
                preprocessing_object=preprocessor,
                trained_model_object=best_model,
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=usvisa_model,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=classification_test_metric,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
