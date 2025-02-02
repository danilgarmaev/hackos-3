from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TrainingArguments, Trainer
from langchain_core.prompts import PromptTemplate
import os
from peft import LoraConfig, get_peft_model
import pandas as pd
from datasets import Dataset


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
                peft_model: str = None,
                lora_r:int = 8,
                lora_alpha:float = 16,
                lora_dropout:float = 0.05,
                lora_bias: str = "none",
                ):
        self.model_name = model_name
        start_time = time.time()
        print(f"Loading model {model_name}...")
        
        tokenizer_path = os.path.join(os.getcwd(), "models", "tokenizers", model_name)
        
        if peft_model and os.path.exists(f"./models/weights/{self.model_name}-{peft_model}-{self.__class__.__name__}"):
            model_path = os.path.join(os.getcwd(), "models", "weights", f"{self.model_name}-{peft_model}-{self.__class__.__name__}")
            self.load_model(model_path, tokenizer_path)
        else:
            model_path = os.path.join(os.getcwd(), "models", "weights", model_name)
            self.load_model(model_path, tokenizer_path)
            
        if prompt_template_path:
            self.prompt_template = PromptTemplate.from_template(open(prompt_template_path, "r").read())
            
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=["q_proj", "v_proj"]
        )
            
        print(f'Model {model_name} loaded in {time.time() - start_time:.2f} seconds.')
        print("------------------------------------------------")
        
    def load_model(self, model_path: str, tokenizer_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def load_dataset(self, dataset_path: str, delimiter="|") -> List[ModelPrediction]:
        df = pd.read_csv(dataset_path, delimiter=delimiter)
    
        df["Prompt"] = df["input"].apply(lambda x: self.prompt_template.invoke({"input": x}).to_string())
        df["Completion"] = df[self.get_prediction_metrics()].apply(lambda row: '|'.join([str(element) for element in row]), axis=1)

        # Convert dataframe to Hugging Face dataset
        dataset = Dataset.from_pandas(df[['Prompt', 'Completion']])

        def tokenize_function(examples):
            # Tokenize the Prompt as input and Completion as target
            inputs = self.tokenizer(examples['Prompt'], padding=True, truncation=True)
            targets = self.tokenizer(examples['Completion'], padding=True, truncation=True)

            # Add labels field for text generation
            inputs['labels'] = targets['input_ids']
            
            return inputs
        
        # Tokenize dataset
        return dataset.map(tokenize_function, batched=True).remove_columns(["Prompt", "Completion"])
    
    def LoRA_training(self, dataset_path: str, delimiter: str = "|") -> AutoModelForCausalLM:
        start_time = time.time()
        print(f"Starting LoRA training...")
        peft_model = get_peft_model(model = self.model, peft_config=self.lora_config)
        print(peft_model.print_trainable_parameters())
        
        training_args = TrainingArguments(
            output_dir=f"./models/weights/{self.model_name}-LoRA-{self.__class__.__name__}",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-4,
            num_train_epochs=3,
            bf16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=4
        )
        
        tokenized_datasets = self.load_dataset(dataset_path, delimiter)
        
        print(tokenized_datasets)
        
        train_dataset, test_dataset = tokenized_datasets.train_test_split(test_size=0.05).values()

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        trainer.train()
        
        print(f'Completion in {time.time() - start_time:.2f} seconds.')
        print("------------------------------------------------")
        
        self.load_model(f"./models/weights/{self.model_name}-LoRA-{self.__class__.__name__}", f"./models/tokenizers/{self.model_name}")
        
        return self.model
        
    @abstractmethod
    def get_prediction_metrics(self) -> List[str]:
        """
        Return a list of keys (strings) representing what metrics the model predicts.
        A list of all available metrics is found in the ModelPrediction dataclass.
        """
        pass

    @abstractmethod
    def predict(self, data: str, verbose: bool = False, max_length: int = 512, temperature: float = 0.4) -> ModelPrediction:
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
    
    def execute(self, prompt: str, verbose: bool, max_length: int, temperature: float) -> str:
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=max_length,
            temperature=temperature,
            truncation = False
            )
        
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

    def predict(self, data: str, verbose: bool = False, max_length: int = 512, temperature: float = 0.4) -> ModelPrediction:
        start_time = time.time()
        print(f"Starting prediction...")
        prompt = self.prompt_template.invoke({"input":data}).to_string()
        
        output = self.execute(prompt, verbose, max_length, temperature)
        
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

    def predict(self, data: dict, verbose: bool = False, max_length: int = 512, temperature: float = 0.4) -> ModelPrediction:
        start_time = time.time()
        print(f"Starting prediction...")
        prompt = self.prompt_template.invoke({"input":data}).to_string()
        
        output = self.execute(prompt, verbose, max_length, temperature)
        
        print(f'Completion in {time.time() - start_time:.2f} seconds.')
        print("------------------------------------------------")
        
        solution = self.text_extraction(output)
            
        return ModelPrediction.from_dict({
            "input": data,
            "error_type": None,
            "severity": None,
            "description": None,
            "solution": solution
        })