import re
from typing import Any, List, Optional, Sequence, Tuple

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.structured_chat.output_parser import (
    StructuredChatOutputParserWithRetries,
)
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.chains.llm import LLMChain
from langchain.tools.render import ToolsRenderer, render_text_description_and_args

#import custom parser
from .HermesOutputParser import HermesAgentOutputParser


def create_hermes_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    tools_renderer: ToolsRenderer = render_text_description_and_args,
) -> Runnable:
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm_with_stop = llm.bind(stop=["Observation"])

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | HermesAgentOutputParser()
    )
    return agent