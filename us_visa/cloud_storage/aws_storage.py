import sys
import os

import boto3
from botocore.exceptions import ClientError

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_object


class SimpleStorageService:
    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            file_objects = [
                file_object.key for file_object in bucket.objects.filter(Prefix=s3_key)
            ]
            return len(file_objects) > 0
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_bucket(self, bucket_name: str):
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True,
    ):
        logging.info("Uploading file to S3")
        try:
            self.s3_resource.meta.client.upload_file(
                from_filename, bucket_name, to_filename
            )
            logging.info(f"Uploaded {from_filename} to s3://{bucket_name}/{to_filename}")
            if remove:
                os.remove(from_filename)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def download_file(self, bucket_name: str, object_name: str, file_name: str):
        logging.info(f"Downloading s3://{bucket_name}/{object_name} to {file_name}")
        try:
            self.s3_resource.Bucket(bucket_name).download_file(object_name, file_name)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def load_model(
        self, model_name: str, bucket_name: str, model_dir: str = None
    ) -> object:
        logging.info("Loading model from S3")
        try:
            tmp_path = model_name if model_dir is None else os.path.join(model_dir, os.path.basename(model_name))
            os.makedirs(os.path.dirname(tmp_path) if os.path.dirname(tmp_path) else ".", exist_ok=True)
            self.download_file(bucket_name, model_name, tmp_path)
            model = load_object(tmp_path)
            logging.info("Model loaded successfully from S3")
            return model
        except Exception as e:
            raise USvisaException(e, sys) from e
