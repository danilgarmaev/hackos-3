from model import Model, ModelPrediction
from langchain_core.prompts import PromptTemplate

class SimpleAgent:
    def __init__(self, model: Model, prompt_template_path: str):
        self.model = model
        self.prompt_template = PromptTemplate.from_template(open(prompt_template_path, "r").read())
        
    def __call__(self, input_data: str) -> ModelPrediction:
        return self.model.predict(self.prompt_template.invoke({"input":input_data}).to_string())