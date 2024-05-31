r"""APIClient is LightRAG'protocol for all API clients, cloud or local to communicate with LightRAG's internal components.

We designed the abstract APIClient class to separate the model API calls from the rest of the system,
making it a plug-and-play component that can be used in functional components like Generator and Embedder.


For a particular API provider, such as OpenAI, we will have a class that inherits from APIClient.
It does four things:

(1) Initialize the client, sync and async.

(2) Convert the standard LightRAG components inputs to the API-specific format.

(3) Call the API and parse the response.

(4) Handle API specific exceptions and errors to retry the call.

Check the subclasses in `components/api_client/` directory for the functional API clients we have.
"""

from typing import Any, Dict, Union, Sequence


from core.component import Component
from core.data_classes import ModelType


API_INPUT_TYPE = Union[
    str, Sequence[str]
]  # str/Sequence for llm and str/sequence/list for embeddings


class APIClient(Component):
    __doc__ = r"""The abstract class for all API clients.

    This interface is designed to bridge the gap between LightRAG components inputs and model APIs.

    You can see examples of the subclasses in components/api_client/ directory.
    """

    def __init__(self, *args, **kwargs) -> None:
        r"""Ensure the subclasses will at least call self._init_sync_client() to initialize the sync client."""
        super().__init__()

        self.sync_client = None
        self.async_client = None

    def _init_sync_client(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_sync_client method"
        )

    def _init_async_client(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_async_client method"
        )

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        r"""Subclass use this to call the API with the sync client.
        model_type: this decides which API, such as chat.completions or embeddings for OpenAI.
        api_kwargs: all the arguments that the API call needs, subclass should implement this method.

        Additionally in subclass you can implement the error handling and retry logic here. See OpenAIClient for example.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _call method")

    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        r"""Subclass use this to call the API with the async client."""
        pass

    def convert_inputs_to_api_kwargs(
        self,
        input: API_INPUT_TYPE = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        r"""
        Bridge the Component's standard input and model_kwargs into API-specific format, the api_kwargs that will be used in _call and _acall methods.

        Args:
            input (API_INPUT_TYPE): user input
            model_kwargs (Dict): model kwargs
            model_type (ModelType): model type

        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _combine_input_and_model_kwargs method"
        )

    def parse_chat_completion(self, completion: Any) -> str:
        r"""
        Parse the chat completion to a structure your sytem standarizes. (here is str)
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_chat_completion method"
        )

    @staticmethod
    def _process_text(text: str) -> str:
        """
        This is specific to OpenAI API, as removing new lines could have better performance in the embedder
        """
        text = text.replace("\n", " ")
        return text

    def _track_usage(self, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
