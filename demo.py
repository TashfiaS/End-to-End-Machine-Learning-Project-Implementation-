# from us_visa.logger import logging
# from us_visa.exception import USVisaException
# import sys


# #logging.info("Logging set up complete. ")

# try:
#     r = 3/0
#     print(r)
# except Exception as e:
#     logging.info(e)
#     raise USVisaException(e, sys) from e

# from dotenv import load_dotenv
# import os

# load_dotenv()

# mongodburl = os.getenv("MONGODB_URL")
# print(mongodburl)

from us_visa.pipeline.training_pipeline import TrainPipeline

pipeline= TrainPipeline()
pipeline.run_pipeline()