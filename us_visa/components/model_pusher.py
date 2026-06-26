import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from us_visa.cloud_storage.aws_storage import SimpleStorageService


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logging.info("Uploading artifacts folder models to S3 bucket")
            s3 = SimpleStorageService()
            s3.upload_file(
                from_filename=self.model_evaluation_artifact.trained_model_path,
                to_filename=self.model_pusher_config.s3_model_key_path,
                bucket_name=self.model_pusher_config.bucket_name,
                remove=False,
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
            )
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
