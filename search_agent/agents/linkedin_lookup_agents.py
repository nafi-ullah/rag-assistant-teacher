from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from tools.tools import get_profile_url_tavily


def lookup(name: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
    )
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                              Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url


if __name__ == "__main__":
    print(lookup(name="Eden Marco Udemy"))


#----------tool ---------------
# But now we'll import another new object which is called a tool.
# What our link chain tools they are Our interfaces that help our link chain.
# Agents, chains or LMS use and interact with the external world.
# So for example, to search online or to search in a database, and you can think about a tool as an
# object that has the following information.
# It has a function to execute a Python function a callable.
# So for example, it may be a function that will be using to search online.
# And it has a description which describes what does this function do and what is the output.
# Which is actually super important when we write the tools because the LM is going to be using that description.



#-----------react agent-----------

# So let's start with the react agents.

# Link chain has implementations for all sorts of agents.

# And the difference between them is the algorithm under the hood that the agent is using in order to

# complete its task.

# Of course, using the tools that we equipped the agent with.

# React is a very famous paper that we discuss in the theory section of this course, which is the most

# popular way to implement an agent with Llms.

# And the create react agent is a function, a built in function in Linkchain, which is going to be receiving

# an LM that will be using to power our agent.

# It's going to receive tools and it's going to receive a prompt, a react prompt.

# It's called and this function is going to return us an agent, which is based on the react algorithm,

# which is using the LM we provided and has the tools we provided it as well.

# This is a sneak peek to the react algorithm, which we're going to cover in depth in the next section

# in this course, but you can simply ignore it for now.

# And I just wanted to put here as a little bit of a teaser.

# And the last object we'll be using is called an agent executor.

# And the agent executor is the runtime of the agent.


#------------agent excecutor------------------
# And the agent executor is the runtime of the agent.

# So this is actually the object which is going to receive our prompts and our instructions what to do.

# And hopefully to finish our task successfully.

# Now, the last object we're going to import is the link chain hub.

# And this is simply a way to download pre-made prompts by the community.