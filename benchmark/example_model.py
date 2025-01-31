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
    is_error: Optional[bool] = Field(description="Whether the line represents an error.")
    error_type: Optional[str] = Field(description="Type of error: see list of categories above.")


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
        _metrics = ["is_error", "error_type"]
        return _metrics

    def predict(self, text: str) -> ModelPrediction:
        _prompt = """
        Given the line from production logs below, identify two things:
        1. is_error: Whether the log represents an error or is normal, as a boolean value (True or False);
        2. error_type: If the log is an error, what type of error it is. Here are your choices:
            ["normal", "warning", "error", "other"]
            Your response for error_type MUST be one of the above. Pick what fits best.
        """
        _output = self.model.get_structured_response(text)
        _pred = ModelPrediction(
            input=text,
            is_error=_output.is_error,
            error_type=_output.error_type,
            # Metrics our model is not benchmarked on:
            event_type=0,  # any integer can go here
            severity="None",  # any string can go here
            root_cause="None",  # any string can go here
            solution="None",  # any string can go here
        )
        return _pred
