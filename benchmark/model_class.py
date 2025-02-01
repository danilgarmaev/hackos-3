from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
import json


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Categories ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are lists of the various error categories your models should be predicting, for
#   their corresponding metrics.
# We cannot properly benchmark your models if they do not produce one of these outputs for
#   these specific metrics ("error_type" and "severity").
_error_types = ["runtime", "fatal", "warning", "no_error"]
_severities = ["notice", "warn", "error"]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ModelPrediction(BaseModel):
    input: Optional[str] = Field(description="Input to the model.")
    error_type: Optional[str] = Field(description="Type of error: see list of categories above.")
    severity: Optional[str] = Field(description="Error severity: see list of categories above.")
    description: Optional[str] = Field(description="Very brief description of possible cause of the error.")
    solution: Optional[str] = Field(description="Very brief outline of solutions to fix the error.")

    def to_dict(self) -> dict:
        _dict = {
            "input": self.input,
            "error_type": self.error_type,
            "severity": self.severity,
            "description": self.description,
            "solution": self.solution,
        }
        return _dict

    @staticmethod
    def from_dict(dictionary: dict):
        return ModelPrediction(**dictionary)


class Model(ABC):
    @abstractmethod
    def get_prediction_metrics(self) -> List[str]:
        """
        Return a list of keys (strings) representing what metrics the model predicts.
        A list of all available metrics is found in the ModelPrediction dataclass.
        """
        # For example, if your model predicts "severity" and "error_type", this function would return:
        #   ["severity", "error_type"]
        pass

    @abstractmethod
    def predict(self, data: str) -> ModelPrediction:
        """
        Return a ModelPrediction containing the model prediction, with keys as mentioned
        in get_prediction_keys().

        Args:
            data: a string representing the line in the log files.

        Returns:
            a ModelPrediction object.
        """
        pass
