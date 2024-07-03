import os
import sqlite3
import json
from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from data_setup import query_db

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

LLAMA_API = os.getenv("LLAMA_API")
LANGSMITH_API = os.getenv("LANGSMITH_API")

model = ChatOpenAI(
    openai_api_key=LLAMA_API,
    openai_api_base="https://api.llama-api.com",
    model="llama3-70b"
)

DB_DESCRIPTION = """You have access to the following tables and columns in a SQLite3 database:

Retail Table
Customer_ID: A unique ID that identifies each customer.
Name: The customer's name.
Gender: The customer's gender: Male, Female.
Age: The customer's age.
Country: The country where the customer resides.
State: The state where the customer resides.
City: The city where the customer resides.
Zip_Code: The zip code where the customer resides.
Product: The product purchased by the customer.
Category: The category of the product.
Price: The price of the product.
Purchase_Date: The date when the purchase was made.
Quantity: The quantity of the product purchased.
Total_Spent: The total amount spent by the customer.
"""

class WorkflowState(TypedDict):
    question: str
    plan: str
    can_answer: bool
    sql_query: str
    sql_result: str
    answer: str

can_answer_router_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \\n

    {data_description} \\n\\n

    Given the user's question, decide whether the question can be answered using the information in the database. \\n\\n

    Return a JSON with two keys, 'reasoning' and 'can_answer', and no preamble or explanation.
    Return one of the following JSON:

    {{"reasoning": "I can find the average total spent by customers in California by averaging the Total_Spent column in the Retail table filtered by State = 'CA'", "can_answer":true}}
    {{"reasoning": "I can find the total quantity of products sold in the Electronics category using the Quantity column in the Retail table filtered by Category = 'Electronics'", "can_answer":true}}
    {{"reasoning": "I can't answer how many customers purchased products last year because the Retail table doesn't contain a year column", "can_answer":false}}

    user
    Question: {question} \\n
    assistant""",
    input_variables=["data_description", "question"],
)

def parse_json_output(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"reasoning": "", "can_answer": False}

def can_answer_router(input_data):
    prompt = can_answer_router_prompt.format(**input_data)
    output = model(prompt)
    return parse_json_output(output)

def check_if_can_answer_question(state):
    result = can_answer_router({"question": state["question"], "data_description": DB_DESCRIPTION})
    return {"plan": result["reasoning"], "can_answer": result["can_answer"]}

def skip_question(state):
    return "no" if state["can_answer"] else "yes"

write_query_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \\n

    {data_description} \\n\\n

    In the previous step, you have prepared the following plan: {plan}

    Return an SQL query with no preamble or explanation. Don't include any markdown characters or quotation marks around the query.
    user
    Question: {question} \\n
    assistant""",
    input_variables=["data_description", "question", "plan"],
)

def parse_string_output(output):
    return output.strip()

def write_query(input_data):
    prompt = write_query_prompt.format(**input_data)
    output = model(prompt)
    return {"sql_query": parse_string_output(output)}

def execute_query(state):
    query = state["sql_query"]
    try:
        return {"sql_result": query_db(query).to_markdown()}
    except Exception as e:
        return {"sql_result": str(e)}

write_answer_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \\n

    In the previous step, you have planned the query as follows: {plan},
    generated the query {sql_query}
    and retrieved the following data:
    {sql_result}

    Return a text answering the user's question using the provided data.
    user
    Question: {question} \\n
    assistant""",
    input_variables=["question", "plan", "sql_query", "sql_result"],
)

def write_answer(input_data):
    prompt = write_answer_prompt.format(**input_data)
    output = model(prompt)
    return {"answer": parse_string_output(output)}

cannot_answer_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \\n

    You cannot answer the user's questions because of the following problem: {problem}.

    Explain the issue to the user and apologize for the inconvenience.
    user
    Question: {question} \\n
    assistant""",
    input_variables=["question", "problem"],
)

def cannot_answer(input_data):
    prompt = cannot_answer_prompt.format(**input_data)
    output = model(prompt)
    return {"answer": parse_string_output(output)}

workflow = StateGraph(WorkflowState)

workflow.add_node("check_if_can_answer_question", check_if_can_answer_question)
workflow.add_node("write_query", write_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("write_answer", write_answer)
workflow.add_node("cannot_answer", cannot_answer)

workflow.set_entry_point("check_if_can_answer_question")

workflow.add_conditional_edges(
    "check_if_can_answer_question",
    skip_question,
    {
        "yes": "cannot_answer",
        "no": "write_query",
    },
)

workflow.add_edge("write_query", "execute_query")
workflow.add_edge("execute_query", "write_answer")
workflow.add_edge("write_answer", END)
workflow.add_edge("cannot_answer", END)

def run_workflow(question):
    state = {"question": question}
    result = workflow.invoke(state)
    return result["answer"]

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer = run_workflow(question)
    print("Answer:", answer)
