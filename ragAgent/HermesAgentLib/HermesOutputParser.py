from __future__ import annotations
import json
import logging
import re
from typing import List, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.agent import AgentOutputParser

logger = logging.getLogger(__name__)


class HermesAgentOutputParser(AgentOutputParser):
    """
    Parses actions encoded in JSON within <tool_call> XML tags.
    Returns a list of AgentAction objects or a single AgentFinish object.

    Recognizes a special action with 'name': 'final_answer' and returns an AgentFinish object for this action.
    If both a final answer and other actions are detected, logs a warning and returns only the AgentFinish object.

    AgentFinish tool calls are performed as such:

    <tool_call>
        {'arguments': {'assistant_reply': 'arbitrary text example: your final reply goes here'}, 'name': 'final_answer'}
    </tool_call>
    """

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        try:
            tool_call_blocks = self._extract_tool_call_blocks(text)
            if not tool_call_blocks:
                raise OutputParserException("No valid <tool_call> blocks found")

            final_action = None
            actions = []
            for block in tool_call_blocks:
                action = self._parse_single_tool_call(block)
                if action:
                    if isinstance(action, AgentFinish):
                        if final_action:
                            logger.warning(
                                "Multiple final_answer actions detected, using the first one."
                            )
                        else:
                            final_action = action
                    else:
                        actions.append(action)

            if final_action:
                if actions:
                    logger.warning(
                        "Final answer detected along with other actions; ignoring other actions."
                    )
                return final_action
            return actions

        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    def _extract_tool_call_blocks(self, text: str) -> List[str]:
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        return re.findall(tool_call_pattern, text, re.DOTALL)

    def _parse_single_tool_call(self, block: str) -> Union[AgentAction, AgentFinish]:
        try:
            # Replace single quotes with double quotes for JSON parsing
            formatted_block = block.replace("'", '"')
            tool_call_data = json.loads(formatted_block)
            action_name = tool_call_data.get('name', '')
            action_input = tool_call_data.get('arguments', {})

            if action_name == 'final_answer':
                return AgentFinish({"output": action_input.get('assistant_reply', '')}, block)
            else:
                return AgentAction(action_name, action_input, block)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in block: {block} - {e}")
            return None

    @property
    def _type(self) -> str:
        return "hermes-agent"
