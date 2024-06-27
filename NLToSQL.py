from langchain.chains import create_sql_query_chain
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline, LlamaTokenizer, LlamaForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
import os
from langchain_community.utilities.sql_database import SQLDatabase
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from operator import itemgetter
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import pandas as pd
from argparse import ArgumentParser
import json
from langchain.memory import ChatMessageHistory
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool


def select_table(desc_path):
    def get_table_details():
        # Read the CSV file into a DataFrame
        table_description = pd.read_csv(desc_path) ##"/teamspace/studios/this_studio/database_table_descriptions.csv"
        table_docs = []

        # Iterate over the DataFrame rows to create Document objects
        table_details = ""
        for index, row in table_description.iterrows():
            table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"

        return table_details
    
    class Table(BaseModel):
        """Table in SQL database."""

        name: str = Field(description="Name of table in SQL database.")
    
    table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
            The tables are:

            {get_table_details()}

            Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

    table_chain = create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)
    
    def get_tables(tables: List[Table]) -> List[str]:
        tables  = [table.name for table in tables]
        return tables

    select_table = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables

    return select_table


def prompt_creation(example_path):

    with open(example_path, 'r') as file: ##'/teamspace/studios/this_studio/few_shot_samples.json'
        data = json.load(file)

    examples = data["examples"]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}\nSQLQuery:"),
            ("ai", "{query}"),
        ]
    )

    vectorstore = Chroma()
    vectorstore.delete_collection()
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        vectorstore,
        k=2,
        input_keys=["input"],
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        input_variables=["input","top_k"],
    )


    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
            few_shot_prompt,
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ]
    )

    print(few_shot_prompt.format(input="How many products are there?"))
        
    return final_prompt



def rephrase_answer():
    answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
    )

    rephrase_answer = answer_prompt | llm | StrOutputParser()

    return rephrase_answer


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--example_path', type=str, default=os.getcwd()+"/few_shot_samples.json")
    parser.add_argument('--desc_path', type=str, default=os.getcwd()+"/database_table_descriptions.csv")
    parser.add_argument('--db_user', type=str, default="root")
    parser.add_argument('--db_password', type=str, default="root")
    parser.add_argument('--db_host', type=str, default="localhost")
    parser.add_argument('--db_name', type=str, default="classicmodels")
    parser.add_argument('--open_ai_key', type=str)
    args = parser.parse_args()

    db_user = args.db_user
    db_password = args.db_password
    db_host = args.db_host
    db_name = args.db_name

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    # print(db.dialect)
    # print(db.get_usable_table_names())
    # print(db.table_info)
    os.environ["OPENAI_API_KEY"] =  args.open_ai_key #"sk-proj-tkqe8k87jGE1AbEYJD7ZT3BlbkFJ5JyTXFxpoycYkciaVNET"

    history = ChatMessageHistory()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    final_prompt = prompt_creation(args.example_path)

    generate_query = create_sql_query_chain(llm, db, final_prompt)

    execute_query = QuerySQLDataBaseTool(db=db)

    chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table(args.desc_path)) |
        RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | execute_query
        )
        | rephrase_answer()
    )

    while True:
        user_input = input("Enter a question for the DB (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        output = chain.invoke({"question": user_input, "messages":history.messages})
        history.add_user_message(user_input)
        history.add_ai_message(output)
        print(output)

    

    

    


