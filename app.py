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
import subprocess
import sys
from transformers import pipeline
import librosa
import soundfile
import datasets
# import sounddevice as sd
import numpy as np
import io
import gradio as gr


model_id = "avnishkanungo/whisper-small-dv"  # update with your model id
pipe = pipeline("automatic-speech-recognition", model=model_id)

def sql_translator(filepath, key):    
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


    def is_ffmpeg_installed():
        try:
            # Run `ffmpeg -version` to check if ffmpeg is installed
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_ffmpeg():
        try:
            if sys.platform.startswith('linux'):
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], check=True)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['/bin/bash', '-c', 'brew install ffmpeg'], check=True)
            elif sys.platform == 'win32':
                print("Please download ffmpeg from https://ffmpeg.org/download.html and install it manually.")
                return False
            else:
                print("Unsupported OS. Please install ffmpeg manually.")
                return False
        except subprocess.CalledProcessError as e:
            print(f"Failed to install ffmpeg: {e}")
            return False
        return True

    def transcribe_speech(filepath):
            output = pipe(
                filepath,
                max_new_tokens=256,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "english",
                },  # update with the language you've fine-tuned on
                chunk_length_s=30,
                batch_size=8,
            )
            return output["text"]
        
    # def record_command():
    #         sample_rate = 16000  # Sample rate in Hz
    #         duration = 8  # Duration in seconds

    #         print("Recording...")

    #         # Record audio
    #         audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
    #         sd.wait()  # Wait until recording is finished

    #         print("Recording finished")

    #         # Convert the audio to a binary stream and save it to a variable
    #         audio_buffer = io.BytesIO()
    #         soundfile.write(audio_buffer, audio, sample_rate, format='WAV')
    #         audio_buffer.seek(0)  # Reset buffer position to the beginning

    #         # The audio file is now saved in audio_buffer
    #         # You can read it again using soundfile or any other audio library
    #         audio_data, sample_rate = soundfile.read(audio_buffer)

    #         # Optional: Save the audio to a file for verification
    #         # with open('recorded_audio.wav', 'wb') as f:
    #         #     f.write(audio_buffer.getbuffer())

    #         print("Audio saved to variable")
    #         return audio_data
    
    def check_libportaudio_installed():
        try:
            # Run `ffmpeg -version` to check if ffmpeg is installed
            subprocess.run(['libportaudio2', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_libportaudio():
        try:
            if sys.platform.startswith('linux'):
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'libportaudio2'], check=True)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['/bin/bash', '-c', 'brew install portaudio'], check=True)
            elif sys.platform == 'win32':
                print("Please download ffmpeg from https://ffmpeg.org/download.html and install it manually.")
                return False
            else:
                print("Unsupported OS. Please install ffmpeg manually.")
                return False
        except subprocess.CalledProcessError as e:
            print(f"Failed to install ffmpeg: {e}")
            return False
        return True

    
    db_user = "admin"
    db_password = "avnishk96"
    db_host = "demo-db.cdm44iseol25.us-east-1.rds.amazonaws.com"
    db_name = "classicmodels"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    # print(db.dialect)
    # print(db.get_usable_table_names())
    # print(db.table_info)
    os.environ["OPENAI_API_KEY"] =  key

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    history = ChatMessageHistory()

    final_prompt = prompt_creation(os.getcwd()+"/few_shot_samples.json")

    generate_query = create_sql_query_chain(llm, db, final_prompt)

    execute_query = QuerySQLDataBaseTool(db=db)

    # if is_ffmpeg_installed():
    #     print("ffmpeg is already installed.")
    # else:
    #     print("ffmpeg is not installed. Installing ffmpeg...")
    # if install_ffmpeg():
    #     print("ffmpeg installation successful.")
    # else:
    #     print("ffmpeg installation failed. Please install it manually.")
    
    # if check_libportaudio_installed():
    #     print("libportaudio is already installed.")
    # else:
    #     print("libportaudio is not installed. Installing ffmpeg...")
    # if install_libportaudio():
    #     print("libportaudio installation successful.")
    # else:
    #     print("libportaudio installation failed. Please install it manually.")

    if os.path.isfile(filepath):
        sql_query = transcribe_speech(filepath)
    else:
        sql_query = filepath
    
    # sql_query = transcribe_speech(filepath)
    chain = (
    RunnablePassthrough.assign(table_names_to_use=select_table(os.getcwd()+"/database_table_descriptions.csv")) |
    RunnablePassthrough.assign(query=generate_query).assign(
    result=itemgetter("query") | execute_query
    )
    | rephrase_answer()
    )

    output = chain.invoke({"question": sql_query, "messages":history.messages})
    history.add_user_message(sql_query)
    history.add_ai_message(output)

    return output


def create_interface():

    demo = gr.Blocks()
    
    mic_transcribe = gr.Interface(
        fn=sql_translator,
        # key_input = gr.Textbox(lines=2, placeholder="Enter text here...", label="Open AI Key"),
        # audio_input = gr.Audio(sources="microphone", type="filepath"),
        inputs = [gr.Audio(sources="microphone", type="filepath"),gr.Textbox(lines=2, placeholder="Enter text here...", label="Open AI Key")],
        outputs=gr.components.Textbox(),
    )

    file_transcribe = gr.Interface(
        fn=sql_translator,
        # key_input = gr.Textbox(lines=2, placeholder="Enter text here...", label="Open AI Key"),
        # query_input = gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text..."),
        inputs = [gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text...") ,gr.Textbox(lines=2, placeholder="Enter text here...", label="Open AI Key")],
        # inputs=gr.Audio(sources="upload", type="filepath"),
        outputs=gr.components.Textbox(),
    )

    with demo:
        gr.TabbedInterface(
            [mic_transcribe, file_transcribe],
            ["Audio Query", "Text Query"],
        )
    
    demo.launch(share=True)

    # return interface


if __name__ == "__main__":
    interface = create_interface()
    # interface.launch(debug=True)