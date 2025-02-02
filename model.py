from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
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
    def __init__(self, 
                model_name: str = "DeepSeek-R1-Distill-Qwen-1.5B", 
                prompt_template_path: str = None,
                max_length: int = 1024,
                temperature: float = 0.4
                ):
        start_time = time.time()
        print(f"Loading model {model_name}...")
        model_path = os.path.join(os.getcwd(), "models", "weights", model_name)
        tokenizer_path = os.path.join(os.getcwd(), "models", "tokenizers", model_name)
        
        self.load_model(model_path, tokenizer_path, max_length, temperature)
        if prompt_template_path:
            self.prompt_template = PromptTemplate.from_template(open(prompt_template_path, "r").read())
        print(f'Model {model_name} loaded in {time.time() - start_time:.2f} seconds.')
        print("------------------------------------------------")
        
    def load_model(self, model_path: str, tokenizer_path: str, max_length: int, temperature: float):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            truncation = False
            )
        
    @abstractmethod
    def get_prediction_metrics(self) -> List[str]:
        """
        Return a list of keys (strings) representing what metrics the model predicts.
        A list of all available metrics is found in the ModelPrediction dataclass.
        """
        pass

    @abstractmethod
    def predict(self, data: str, verbose: bool = False) -> ModelPrediction:
        """
        Return a ModelPrediction containing the model prediction, with keys as mentioned
        in get_prediction_keys().

        Args:
            data: a string representing the line in the log files.

        Returns:
            a ModelPrediction object.
        """
        pass
    
    def getConfig(self) -> dict:
        return self.model.config.to_dict()
    
    def execute(self, prompt: str, verbose: bool) -> str:
        output = self.generator(prompt)[0]["generated_text"]
        if verbose:
            print(output)
            
        return output.split("###START_SESSION###\n")[-1].split("\n###END_SESSION###")[0]
    
    def json_extraction(self, output: str) -> dict:
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        json_str = output[json_start:json_end]
        
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            json_data = {}
            
        return json_data
    
    def text_extraction(self, output: str) -> str:
        return output.split("Answer:\n")[1]
        
class Problem_Model(Model):        
    def get_prediction_metrics(self) -> List[str]:
        return ["severity", "error_type", "description"]

    def predict(self, data: str, verbose: bool = False) -> ModelPrediction:
        start_time = time.time()
        print(f"Starting prediction...")
        prompt = self.prompt_template.invoke({"input":data}).to_string()
        
        output = self.execute(prompt, verbose)
        
        print(f'Completion in {time.time() - start_time:.2f} seconds.')
        print("------------------------------------------------")
        
        json_data = self.json_extraction(output)
            
        return ModelPrediction.from_dict({
            "input": data,
            "error_type": json_data["error_type"],
            "severity": json_data["severity"],
            "description": json_data["description"],
            "solution": None
        })
        
class Solution_Model(Model):        
    def get_prediction_metrics(self) -> List[str]:
        return ["solution"]

    def predict(self, data: dict, verbose: bool = False) -> ModelPrediction:
        start_time = time.time()
        print(f"Starting prediction...")
        prompt = self.prompt_template.invoke({"error_type":data["error_type"], "severity": data["severity"], "description": data["description"]}).to_string()
        
        output = self.execute(prompt, verbose)
        
        print(f'Completion in {time.time() - start_time:.2f} seconds.')
        print("------------------------------------------------")
        
        solution = self.text_extraction(output)
            
        return ModelPrediction.from_dict({
            "input": data["input"],
            "error_type": data["error_type"],
            "severity": data["severity"],
            "description": data["description"],
            "solution": solution
        })