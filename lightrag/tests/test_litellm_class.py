import unittest
import os
from unittest.mock import patch
from lightrag.core.types import ModelType
from lightrag.components.model_client import LiteLLMClient
from lightrag.core.generator import Generator
from lightrag.core.embedder import Embedder

from lightrag.utils import setup_env  # ensure you have .env with OPENAI_API_KEY

# need to setup env


class TestEnvironment(unittest.TestCase):

    @patch.dict(os.environ, clear=True)
    def test_no_api_key_raises_error(self):
        with self.assertRaises(ValueError):
            client = LiteLLMClient()
            client.call({"model": "gpt-4o"}, ModelType.LLM)

    def test_no_model_logs_error(self):
        with self.assertRaises(ValueError):
            generate = Generator(
                model_client=LiteLLMClient(), model_kwargs={}
            )
            generate.call({"input_str": "Hello World"})

    @patch.dict(os.environ, clear=True)
    def test_generator_no_api_key_logs_error(self):
        with self.assertRaises(ValueError):
            generate = Generator(
                model_client=LiteLLMClient(), model_kwargs={"model": "gpt-4o"}
            )
            generate.call({"input_str": "Hello World"})

    @patch.dict(os.environ, clear=True)
    def test_embedder_no_api_key_call_logs_error(self):
        # Should this raise a value error or should we send the output back to be handled by the developer?
        with self.assertRaises(ValueError):
            embed = Embedder(
                model_client=LiteLLMClient(),
                model_kwargs={"model": "text-embedding-ada-002"},
            )
            embed.call(input="Hello World")
            
    @patch.dict(os.environ, clear=True)
    async def test_embedder_no_api_key_acall_logs_error(self):
        with self.assertRaises(ValueError):
            embed = Embedder(
                model_client=LiteLLMClient(),
                model_kwargs={"model": "text-embedding-ada-002"},
            )
            await embed.acall(input="Hello World")


if __name__ == "__main__":
    unittest.main()
