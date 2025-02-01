from .model_class import Model, ModelPrediction

import os
from dataclasses import dataclass
import getpass
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


@dataclass
class LanguageModelConfig:
    model: str = "openai"
    model_name: str = "gpt-4o"
    temperature: float = 0
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    max_retries: int = 1


class GPTOutput(BaseModel):
    error_type: Optional[str] = Field(description="Type of error: see list of categories above.")
    severity: Optional[str] = Field(description="Severity of the error: one of ['error', 'warn', 'notice']")
    description: Optional[str] = Field(description="One-line specific description of the log line.")
    solution: Optional[str] = Field(description="One-line specific solution to the log line, if error or warning.")


class LanguageModel:
    def __init__(self, config: LanguageModelConfig, structured_output = None):
        self.config = config
        self.structured_output = structured_output

        # Initialize the underlying LLM using LangChain:
        _supported_models = ["openai"]
        self.model = None
        self.structured_model = None
        assert self.config.model in _supported_models, f"Model f{self.config.model} not yet supported."
        if self.config.model == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
            self.model = ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )

        # Add structure if applicable:
        if structured_output is not None and self.structured_model is None:
            self.structured_model = self.model.with_structured_output(structured_output)

    def get_structured_response(self, system_prompt: str):
        """
        Invokes the structured LLM using the provided prompt(s).
        """
        assert self.structured_model is not None, "Structured model has not been defined."
        output = self.structured_model.invoke(system_prompt)
        return output


class GPTModel(Model):
    def __init__(self, config: LanguageModelConfig = LanguageModelConfig()):
        super().__init__()
        self.model = LanguageModel(config, structured_output=GPTOutput)

    def get_prediction_metrics(self):
        _metrics = ["error_type", "severity", "description", "solution"]
        return _metrics

    def predict(self, text: str) -> ModelPrediction:
        _prompt = """
        Given the line from production logs below, identify these things:
        1. severity: Severity of the error, must be one of ["notice", "warn", "error"];
        2. error_type: If the log is an error, what type of error it is. Here are your choices:
            ["fatal", "runtime", "no_error", "warning"]
            Your response for error_type MUST be one of the above. Pick what fits best.
        3. description: One-line description of the log line.
        4. solution: One-line description of solutions, if the log line is an error or warning.
        """
        _output = self.model.get_structured_response(f"{_prompt}\n\n{text}")
        print(_output)
        _pred = ModelPrediction(
            input=text,
            error_type=_output.error_type,
            severity=_output.severity,
            description=_output.description,  # any string can go here
            solution=_output.solution,  # any string can go here
        )
        return _pred
