from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from search import search
from dotenv import dotenv_values
from langchain.tools import tool
import json
from langchain.prompts import PromptTemplate
from embeddings import create_embeddings

config=dotenv_values(".env")

@tool
def get_info(tool_input: str) -> str:
    """Searches for information in a given file based on a query. Input should be a JSON string with 'query' and 'file_path' keys."""
    try:
        parsed_input = json.loads(tool_input)
        query = parsed_input["query"]
        file_path = parsed_input["file_path"]
    except (json.JSONDecodeError, KeyError) as e:
        return f"Error parsing tool input: {e}. Input was: {tool_input}"
    
    results = [match['metadata'] for match in search(query, file_path).matches]
    return "This is the information for the query: " + query + " from the file: " + file_path + " " + str(results)

prompt_template = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:
{tools}

Your purpose is to find information in the file and return it to the user.

If you need to create embeddings, you can use the create_embeddings tool.
If you need to search the file, you can use the get_info tool.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one or more of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Chat History:
{chat_history}
Question: {input}
{agent_scratchpad}
""")
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=config["OPENAI_API_KEY"]
)    
tools = [get_info, create_embeddings]

agent = create_react_agent(llm, tools,prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# Use with chat history
from langchain_core.messages import AIMessage, HumanMessage

def _format_chat_history(chat_history: list) -> str:
    formatted_history = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_history.append(f"{message.content}")
        elif isinstance(message, AIMessage):
            formatted_history.append(f"{message.content}")
    return "\n".join(formatted_history)

chat_history_list = [
    AIMessage(content="Hello! How can I assist you today?")
]

'''
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    chat_history_list.append(HumanMessage(content=user_input))
    formatted_chat_history = _format_chat_history(chat_history_list)

    response = agent_executor.invoke(
        {
            "input": user_input,
            "chat_history": formatted_chat_history,
        }
    )
    agent_response = response["output"]
    print(f"Agent: {agent_response}")
    chat_history_list.append(AIMessage(content=agent_response))
'''