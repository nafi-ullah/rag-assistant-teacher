# Importing necessary modules
from typing import Union, List  # For type hinting
import re  # Regular expressions (not used here explicitly)
from dotenv import load_dotenv  # Load environment variables
from langchain.agents import tool  # Decorator to define tools for the agent
from langchain.agents.format_scratchpad import format_log_to_str  # Formatting agent scratchpad logs
from langchain.agents.output_parsers import ReActSingleInputOutputParser  # Parses LLM output into actions
from langchain_openai import ChatOpenAI  # OpenAI LLM model for generating responses
from langchain.prompts import PromptTemplate  # For formatting the prompt dynamically
from langchain.schema import AgentAction, AgentFinish  # Two types of outputs from the agent
from langchain.tools import Tool  # Base tool class for defining custom tools
from langchain.tools.render import render_text_description  # Render tools for display

from callbacks import AgentCallbackHandler  # Custom callback handler for logging (User-defined)

# Load environment variables (for API keys, etc.)
load_dotenv()

# Tool function: A tool is like a helper function that the agent can use to process data
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")  # Debugging log
    text = text.strip("'\n").strip(
        '"'
    )  # Remove unnecessary quotes and newline characters
    return len(text)  # Return the length of the text

# Function to find a tool by its name in a list of tools
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool  # Return the tool if found
    raise ValueError(f"Tool with name {tool_name} not found")  # Raise error if tool not found

# Main execution block
if __name__ == "__main__":
    print("Hello ReAct LangChain!")  # Just a startup message
    
    # Define available tools (in this case, just one tool)
    tools = [get_text_length]
    
    # Define the prompt template to guide the agent's response format
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """
    
    # Creating a prompt using the template
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),  # Render tool descriptions dynamically
        tool_names=", ".join([t.name for t in tools]),  # Extract tool names as a comma-separated string
    )
    
    # Initialize the OpenAI language model (ChatOpenAI)
    llm = ChatOpenAI(
        temperature=0,  # Set to 0 for deterministic output (no randomness)
        stop=["\nObservation", "Observation"],  # Stop conditions for parsing agent output
        callbacks=[AgentCallbackHandler()],  # Custom callback handler for debugging/logging
    )
    
    intermediate_steps = []  # Stores intermediate steps of the agent's reasoning
    
    # Define the agent pipeline using a series of functional transformations
    agent = (
        {
            "input": lambda x: x["input"],  # Extract user input
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),  # Format agent history
        }
        | prompt  # Apply the prompt template
        | llm  # Process with the language model (ChatOpenAI)
        | ReActSingleInputOutputParser()  # Parse the output to determine next action
    )
    
    # Agent execution loop
    agent_step = ""  # Initial agent step
    while not isinstance(agent_step, AgentFinish):  # Loop until the agent reaches a final answer
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the word: DOG",  # Sample question
                "agent_scratchpad": intermediate_steps,  # Provide intermediate reasoning steps
            }
        )
        print(agent_step)  # Print the step result
        
        if isinstance(agent_step, AgentAction):  # If the agent decides to take an action
            tool_name = agent_step.tool  # Extract the tool name
            tool_to_use = find_tool_by_name(tools, tool_name)  # Find the tool from available tools
            tool_input = agent_step.tool_input  # Extract tool input
            
            # Execute the selected tool with the given input
            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")  # Print observation result
            
            # Store this step for the next iteration
            intermediate_steps.append((agent_step, str(observation)))
    
    # If the agent has reached a final answer, print it
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)  # Output the final answer