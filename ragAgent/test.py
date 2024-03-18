from langchain import hub
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import Ollama

# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from HermesAgentLib import HermesAgent

# Create the language model
model = Ollama(
    model="hermes2proV2", verbose=True, stop=["<|im_end|>", "Observation:", "<Observation>"]
)  # without defined system prompt template

# tools


@tool
def search_test_one(query: str) -> str:
    """get test string one"""
    return "test string: LangChain; Your query: " + query


@tool
def search_test_two(query: str) -> str:
    """get test string two"""
    return "test string: Ollama; Your query: " + query


@tool
def search_test_three(*, query: str) -> str:
    """Get test string three"""
    return "test string: Hermes; Your query: " + query


# Create the list of tools
tools = [search_test_one, search_test_two, search_test_three]

SYSTEM_PROMPT_TEMPLATE = """
system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. 
Don't make assumptions about what values to plug into functions. Here are the available tools:

<tools>
{tools}
</tools>

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{{'arguments': <args-dict>, 'name': <tool-name>}}
</tool_call>

<tool-name> can only be one of: {tool_names}

Reminder to ALWAYS respond with at least one valid <tool_call></tool_call block>!
If you decide that no use of any above-mentioned tool is necessary, or have received all the context you need, use the "final_answer" tool to send your reply!
It is used as follows:

<tool_call>
{{'arguments': {{'assistant_reply': 'arbitrary text example: your final reply goes here'}}, 'name': 'final_answer'}}
</tool_call>

Use tools if necessary.
When you have defined all tool calls, type "<Observation>" or end your train of thought to be notified of their outputs
"""

HUMAN_PROMPT_TEMPLATE = "<|im_start|>user Question: {input} <|im_end|>"

AI_PROMPT_TEMPLATE = """<|im_start|>assistant 
                        {agent_scratchpad}
                        """

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            input_variables=["tool_names", "tools"], template=SYSTEM_PROMPT_TEMPLATE
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template(
            input_variables=["input"], template=HUMAN_PROMPT_TEMPLATE
        ),
        AIMessagePromptTemplate.from_template(
            input_variables=["agent_scratchpad"], template=AI_PROMPT_TEMPLATE
        ),
    ]
)

print(f"Prompt: {prompt}")

# Create the memory object
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=3, return_messages=True
)

# Construct the JSON agent
agent = HermesAgent.create_hermes_agent(model, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
)

query = (
    "this is a test. Please use your 'search_test_one' tool!"
)

# Run the agent with the input
agent_response = agent_executor.invoke({"input": query})

memory_content = memory.buffer
print(f"Memory content: {memory_content}")

# Return the agent response and memory contents
print(agent_response)
