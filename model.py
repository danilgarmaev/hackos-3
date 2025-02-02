from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os


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
        
class DeepSeek_1_5B_Model(Model):    
    def __init__(self, model_name: str = "DeepSeek-R1-Distill-Qwen-1.5B"):
        start_time = time.time()
        print(f"Loading model {model_name}...")
        model_path = os.path.join(os.getcwd(), "models", "weights", model_name)
        tokenizer_path = os.path.join(os.getcwd(), "models", "tokenizers", model_name)
        
        self.load_model(model_path, tokenizer_path)
        print(f'Model {model_name} loaded in {time.time() - start_time:.2f} seconds.')
        print("------------------------------------------------")

    def load_model(self, model_path: str, tokenizer_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=tokenizer)
        
    def getConfig(self) -> dict:
        return self.model.config.to_dict()
    
    def get_prediction_metrics(self) -> List[str]:
        return ["severity", "error_type", "description"]

    def predict(self, data: str) -> ModelPrediction:
        start_time = time.time()
        print(f"Starting prediction...")
        
        # Tokenize the input
        output = self.generator(
            data,
            max_length=1024,
            temperature=0.4,
            truncation=False,
        )[0]["generated_text"]
        
        print(f'Completion in {time.time() - start_time:.2f} seconds.')
        print("------------------------------------------------")
        output = output.split("###START_SESSION###\n")[2].split("###END_SESSION###\n")[0]
        print(f"{output}")
        
        # Extract JSON from the output
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        json_str = output[json_start:json_end]
        
        # Parse the JSON string
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            json_data = {}
            
        return ModelPrediction.from_dict({
            "input": output.split("\n\n# Thinking process")[0].replace("# Prompt\n", ""),
            "error_type": json_data["error_type"],
            "severity": json_data["severity"],
            "description": json_data["description"],
            "solution": None
        })
        