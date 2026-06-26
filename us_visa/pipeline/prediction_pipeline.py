import sys
import pandas as pd

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.constants import CURRENT_YEAR, SAVED_MODEL_PATH
from us_visa.utils.main_utils import load_object
from us_visa.ml.model.estimator import TargetValueMapping


class USvisaData:
    """Holds a single prediction record and converts it to a DataFrame."""

    def __init__(
        self,
        continent: str,
        education_of_employee: str,
        has_job_experience: str,
        requires_job_training: str,
        no_of_employees: int,
        yr_of_estab: int,
        region_of_employment: str,
        prevailing_wage: float,
        unit_of_wage: str,
        full_time_position: str,
    ):
        self.continent = continent
        self.education_of_employee = education_of_employee
        self.has_job_experience = has_job_experience
        self.requires_job_training = requires_job_training
        self.no_of_employees = no_of_employees
        self.yr_of_estab = yr_of_estab
        self.region_of_employment = region_of_employment
        self.prevailing_wage = prevailing_wage
        self.unit_of_wage = unit_of_wage
        self.full_time_position = full_time_position

    def get_usvisa_input_data_frame(self) -> pd.DataFrame:
        try:
            company_age = CURRENT_YEAR - int(self.yr_of_estab)
            data = {
                "no_of_employees": [self.no_of_employees],
                "company_age": [company_age],
                "prevailing_wage": [self.prevailing_wage],
                "education_of_employee": [self.education_of_employee],
                "continent": [self.continent],
                "region_of_employment": [self.region_of_employment],
                "unit_of_wage": [self.unit_of_wage],
                "has_job_experience": [1 if self.has_job_experience == "Y" else 0],
                "requires_job_training": [1 if self.requires_job_training == "Y" else 0],
                "full_time_position": [1 if self.full_time_position == "Y" else 0],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise USvisaException(e, sys) from e


class USvisaClassifier:
    def __init__(self):
        try:
            self.model = load_object(file_path=SAVED_MODEL_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> str:
        try:
            result = self.model.predict(dataframe)
            return TargetValueMapping().reverse_mapping()[result[0]]
        except Exception as e:
            raise USvisaException(e, sys) from e
