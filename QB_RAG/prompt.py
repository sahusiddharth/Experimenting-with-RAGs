import json
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue as BasePromptValue
from pydantic import BaseModel, model_validator
import typing as t

Example = t.Dict[str, t.Any]


class PromptValue(BasePromptValue):
    prompt_str: str

    def to_messages(self) -> t.List[BaseMessage]:
        """Return prompt as a list of Messages."""
        return [HumanMessage(content=self.to_string())]

    def to_string(self) -> str:
        return self.prompt_str


class Prompt(BaseModel):
    """
    Prompt is a class that represents a prompt for the ragas metrics.

    Attributes:
        instruction (str): The instruction for the prompt.
        output_format_instruction (str): The output format instruction for the prompt.
        examples (List[Dict[str, Any]]): List of example inputs and outputs for the prompt.
        input_keys (List[str]): List of input variable names.
        output_key (str): The output variable name.
        output_type (Literal["json", "str"]): The type of the output (default: "json").
        language (str): The language of the prompt (default: "english").
    """

    instruction: str
    output_format_instruction: str = ""
    examples: t.List[Example] = []
    input_keys: t.List[str] = [""]
    output_key: str = ""
    output_type: t.Literal["json", "str"] = "json"
    language: str = "english"

    @model_validator(mode='after')
    def validate_prompt(self) -> 'Prompt':
        """
        Validate the template string to ensure that it is in desired format.
        """
        if not self.instruction:
            raise ValueError("instruction cannot be empty")
        if not self.input_keys:
            raise ValueError("input_keys cannot be empty")
        if not self.output_key:
            raise ValueError("output_key cannot be empty")

        if self.examples:
            for no, example in enumerate(self.examples):
                for inp_key in self.input_keys:
                    if inp_key not in example:
                        raise ValueError(
                            f"example {no+1} does not have the variable {inp_key} in the definition"
                        )
                if self.output_key not in example:
                    raise ValueError(
                        f"example {no+1} does not have the variable {self.output_key} in the definition"
                    )
                if self.output_type.lower() == "json":
                    try:
                        if self.output_key in example:
                            if isinstance(example[self.output_key], str):
                                json.loads(example[self.output_key])
                    except ValueError as e:
                        raise ValueError(
                            f"{self.output_key} in example {no+1} is not in valid json format: {e}"
                        )
        return self

    def to_string(self) -> str:
        """
        Generate the prompt string from the variables.
        """
        prompt_elements = [self.instruction]
        if self.output_format_instruction:
            prompt_elements.append(
                "\n"
                + self.output_format_instruction.replace("{", "{{").replace("}", "}}")
            )
        prompt_str = "\n".join(prompt_elements) + "\n"

        if self.examples:
            prompt_str += "\nExamples:\n"
            # Format the examples to match the Langchain prompt template
            for example in self.examples:
                for key, value in example.items():
                    is_json = isinstance(value, (dict, list))
                    value = (
                        json.dumps(value, ensure_ascii=False).encode("utf8").decode()
                    )
                    value = (
                        value.replace("{", "{{").replace("}", "}}")
                        if self.output_type.lower() == "json"
                        else value
                    )
                    prompt_str += (
                        f"\n{key}: {value}"
                        if not is_json
                        else f"\n{key}: ```{value}```"
                    )
                prompt_str += "\n"

        prompt_str += "\nYour actual task:\n"

        if self.input_keys:
            prompt_str += "".join(f"\n{key}: {{{key}}}" for key in self.input_keys)
        if self.output_key:
            prompt_str += f"\n{self.output_key}: \n"

        return prompt_str

    def format(self, **kwargs: t.Any) -> PromptValue:
        """
        Format the Prompt object into a ChatPromptTemplate object to be used in metrics.
        """
        if set(self.input_keys) != set(kwargs.keys()):
            raise ValueError(
                f"Input variables {self.input_keys} do not match with the given parameters {list(kwargs.keys())}"
            )
        for key, value in kwargs.items():
            if isinstance(value, str):
                kwargs[key] = json.dumps(value)

        prompt = self.to_string()
        return PromptValue(prompt_str=prompt.format(**kwargs))
