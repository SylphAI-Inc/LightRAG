"""Implementation of TextGrad: Automatic “Differentiation” via Text"""

from typing import Union

# if TYPE_CHECKING:
from lightrag.core.generator import Generator


from lightrag.core.component import Component
from lightrag.core import ModelClient
from ..parameter import Parameter
from typing import Dict
from copy import deepcopy
import logging

log = logging.getLogger(__name__)


TEXT_LOSS_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
{{eval_system_prompt}}
<END_OF_SYSTEM_PROMPT>
<USER>
{{eval_user_prompt}}
</USER>
"""


# TODO: limit to one engine
# TODO: this can be merged with its numerical eval_function
class LLMAsTextLoss(Component):
    __doc__ = r"""Evaluate the final RAG response using an LLM judge.

    The LLM judge will have:
    - eval_system_prompt: The system prompt to evaluate the response.
    - y_hat: The response to evaluate.
    - Optional: y: The correct response to compare against.

    The loss will be a Parameter with the evaluation result and can be used to compute gradients.
    This loss use LLM/Generator as the computation/transformation operator, so it's gradient will be
    found from the Generator's backward method.
    """

    def __init__(
        self,
        prompt_kwargs: Dict[str, Union[str, Parameter]],
        model_client: ModelClient,
        model_kwargs: Dict[str, object],
        engine: Generator = None,
    ):
        super().__init__()
        prompt_kwargs = deepcopy(prompt_kwargs)
        for key, value in prompt_kwargs.items():
            if isinstance(value, str):
                prompt_kwargs[key] = Parameter(
                    data=value, requires_opt=False, role_desc=key  # TODO: role_desc
                )
        self.prompt_kwargs = prompt_kwargs
        # this is llm as judge (loss) to get the loss
        self.loss_llm = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=TEXT_LOSS_TEMPLATE,
            prompt_kwargs=prompt_kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Parameter:

        return self.loss_llm.forward(*args, **kwargs)