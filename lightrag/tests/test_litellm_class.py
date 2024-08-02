import unittest
from unittest.mock import patch
from lightrag.core.types import ModelType
from lightrag.components.model_client import LiteLLMClient
from lightrag.core.generator import Generator
from lightrag.core.embedder import Embedder

from lightrag.utils import setup_env  # ensure you have .env with OPENAI_API_KEY

# need to setup env


class TestEnvironment(unittest.TestCase):
    
    def test_no_api_key_raises_error(self):
        with self.assertRaises(ValueError):
            client = LiteLLMClient()
            client.call({"model":"gpt-4o"}, ModelType.LLM)
    
    def test_generator_no_api_key_logs_error(self):
        with self.assertLogs(level="ERROR"):
            generate = Generator(model_client=LiteLLMClient(), model_kwargs={"model": "gpt-4o"})
            generate.call({"input_str": "Hello World"})
    
    def test_embedder_no_api_key_logs_error(self):
        # Should this raise a value error or should we send the output back to be handled by the developer?
        with self.assertRaises(ValueError):
            embed = Embedder(model_client=LiteLLMClient(), model_kwargs={"model": "text-embedding-ada-002"})
            embed.call(input="Hello World")

if __name__ == "__main__":
    unittest.main()
