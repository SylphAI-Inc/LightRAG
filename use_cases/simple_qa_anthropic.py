"""
We just need to very basic generator that can be used to generate text from a prompt.
"""

from core.generator import Generator
from core.component import Component

from components.api_client import AnthropicAPIClient

import utils.setup_env


class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        self.generator = Generator(
            model_client=AnthropicAPIClient,
            model_kwargs={"model": "claude-3-opus-20240229", "max_tokens": 1000},
            preset_prompt_kwargs={
                "task_desc_str": "You are a helpful assistant and with a great sense of humor."
            },
        )

    def call(self, query: str) -> str:
        return self.generator.call(input=query)


if __name__ == "__main__":
    simple_qa = SimpleQA()
    print(simple_qa)
    print("show the system prompt")
    simple_qa.generator.print_prompt()
    print(simple_qa.call("What is the capital of France?"))