"""Meta-prompts for the backward engine.
Adapted from TextGrad: Automatic “Differentiation” via Text."""

GLOSSARY_TEXT_BACKWARD = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.
# - <VARIABLE>: Specifies the span of the variable.
# - <ROLE>: The role description of the variable."""

### Backward engine: system prompt

# TODO: maybe it can propose new values.

BACKWARD_SYSTEM_PROMPT = r"""{#Overall optimizer description#}
You are the feedback(gradient) engine in a larger optimization system.

Your goal is to give feedback to a variable in <VARIABLE> tags in order to improve the objective specified in <OBJECTIVE_FUNCTION> tags.
The variable may be solution to problems, prompt/instruction to langage model, code, or any other text-based variable.
{#Task specifics#}
Remember:
- Pay attention to the role description of the variable, and the context in which it is used.
- You should assume that the variable will be used in a similar context in the future.
{#- Only provide strategies, explanations, and methods to change in the variable.#}
- DO NOT propose a new version of the variable, that will be the job of the optimizer.
- Your only job is to send feedback and criticism (compute 'gradients').
    For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.
- If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should respond with "It works well in this case, no critical feedback."
- BE CONCISE, CRITICAL, and CREATIVE.
"""

###  Backward engine: user prompt

# First part to provide context of LLM as gradFunction

CONVERSATION_TEMPLATE = r"""<LM_PROMPT> {{llm_prompt}} </LM_PROMPT>
<LM_OUTPUT> {{response_value}} </LM_OUTPUT>"""


# When the parameter has no gradient, it is the start of the backpropagation chain, used as a loss function
CONVERSATION_START_INSTRUCTION_BASE = r"""
<START_OF_VARIABLE_DESC>
Variable type: <TYPE>{{param_type}}</TYPE>
Variable value: <VARIABLE> {{variable_value}} </VARIABLE>
Role Description: <ROLE>{{variable_desc}}</ROLE>.
{% if instruction_to_backward_engine %}
Note: {{instruction_to_backward_engine}}
{% endif %}
<END_OF_VARIABLE_DESC>

Here is an evaluation of the variable using a language model:
{{conversation_str}}
"""

# When the parameter has a gradient, it is the continuation of the backpropagation chain, a layer in the models
CONVERSATION_START_INSTRUCTION_CHAIN = r"""
<START_OF_VARIABLE_DESC>
Variable type: <TYPE>{{param_type}}</TYPE>
Variable value: <VARIABLE> {{variable_value}} </VARIABLE>
Role Description: <ROLE>{{variable_desc}}</ROLE>.
{% if instruction_to_backward_engine %}
Note: {{instruction_to_backward_engine}}
{% endif %}
<END_OF_VARIABLE_DESC>

Here is a conversation with a language model (LM):
{{conversation_str}}
"""

# Second part of the user prompt
OBJECTIVE_INSTRUCTION_BASE = r"""<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output.
Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>"""


OBJECTIVE_INSTRUCTION_CHAIN = r"""This conversation is part of a larger system. The <LM_OUTPUT> was later used as {{response_desc}}.
<OBJECTIVE_FUNCTION>Your goal is to give feedback to the variable to address the following feedback on the LM_OUTPUT: {{response_gradient}} </OBJECTIVE_FUNCTION>"""


# Third part pf the user prompt


# EVALUATE_VARIABLE_INSTRUCTION = r"""We are interested in giving feedback to the {{variable_desc}}
# for this conversation. Specifically, give feedback to the following span of text:
# <VARIABLE> {{variable_short}} </VARIABLE>
# Given the above history, describe how the {{variable_desc}}
# could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.
# """

# The full backward engine prompt template
FEEDBACK_ENGINE_TEMPLATE = r"""<START_OF_SYSTEM_PROMPT>
You are the feedback(gradient) engine in a larger optimization system.

Your goal is to give feedback to a variable in <VARIABLE> tags in order to improve the objective specified in <OBJECTIVE_FUNCTION> tags.
The variable may be solution to problems, prompt/instruction to langage model, code, or any other text-based variable.
{#Task specifics#}
Remember:
- Pay attention to the role description of the variable, and the context in which it is used.
- You should assume that the variable will be used in a similar context in the future.
{#- Only provide strategies, explanations, and methods to change in the variable.#}
- DO NOT propose a new version of the variable, that will be the job of the optimizer.
- Your only job is to send feedback and criticism (compute 'gradients').
    For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.
- If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should respond with "It works well in this case, no critical feedback."
- BE CONCISE, CRITICAL, and CREATIVE.
<END_OF_SYSTEM_PROMPT>
<START_OF_USER_PROMPT>
{{ "\"\"\"" }}{{conversation_sec}}{{ "\"\"\"" }}

{{objective_instruction_sec}}

<END_OF_USER_PROMPT>"""


# TODO: Not fully sure about the function of this template
GRADIENT_TEMPLATE = r"""Here is a conversation:
<CONVERSATION>{{context}}</CONVERSATION>
This conversation is potentially part of a larger system. The output is used as {{response_desc}}
Here is the feedback we got for {{variable_desc}} in the conversation:
    <FEEDBACK>{{feedback}}</FEEDBACK>"""
