"""
We just need to very basic generator that can be used to generate text from a prompt.
"""

from core.openai_llm import OpenAIGenerator

from core.component import Component

# TODO: make the environment variable loading more robust, and let users specify the .env path
import dotenv

dotenv.load_dotenv()


class SimpleQA(Component):
    def __init__(self):
        super().__init__()
        model_kwargs = {"model": "gpt-3.5-turbo"}
        self.generator = OpenAIGenerator(model_kwargs=model_kwargs)
        self.generator.print_prompt()

    def call(self, query: str) -> str:
        return self.generator.call(input=query)


if __name__ == "__main__":
    simple_qa = SimpleQA()
    print(simple_qa)
    print(simple_qa.call("What is the capital of France?"))