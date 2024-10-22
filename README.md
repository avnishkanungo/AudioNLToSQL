# AudioNLToSQL

- This project aims to convert Natural Language inputs in form of text and audio to SQL Queries leveraging LLMs and prompt engineering and automatically running these queries and providing an output from the DB, for the same we are leveraging Chat GPT 3.5 Turbo, Langchain and Huggingface Transformers for the implementation. 
- We have also finetuned the performance by providing a way to select only the required tables for the prompts, using a csv which contains the description of the database, for the same we have used a vector database for semantic matching to choose relevant tables as per the table descriptions provided. Refer to the database_table_descriptions.csv file.
- And have leveraged few shot learning method provided for openAI models via langchain to train the model to the pecific kind of queries that we need answered. Refer to the few_shot_samples.json file.
- For the audio module, I trained a Automatic Speech Recognition model using hugingface transformers saved the same to the HuggingFace Hub and used it to convert recorded audio into text which is then passed as input to the Langchain LLM setup to convert to SQL queries which are run on the database and outputs from the same are returned.
Model link on HuggingFace: https://huggingface.co/avnishkanungo/whisper-small-dv


This script can be run on any local msql database provided you input the correct username, password, host name and DB name. All of these can be passed as arguments to the command to run the provided script.

**To install required libraries:**
```
pip install -r requirements.txt
```

**To run this code please traverse into the directory where the NLT0SQL.py script is present and use run the below command with your Open AI API key on your own database please use the below command:
**
```
python3 NLToSQL.py --db_user "SQL_DB_Username" --db_password "PASSWORD" --db_host "HOSTNAME" --db_name "DATABSE_NAME" --open_ai_key "YOUR_OPEN_AI_API_KEY"
```

**You can make required changes to the the database_table_descriptions.csv and few_shot_samples.json files as per your requirement. If need be you can use your own files for the same too, please refer to the below example command:**

```
python3 NLToSQL.py --desc_path "PATH_FOR_DATABASE_DESCRIPTION" --example_path "few_shot_examples_path" --db_user "SQL_DB_Username" --db_password "PASSWORD" --db_host "HOSTNAME" --db_name "DATABSE_NAME" --open_ai_key "YOUR_OPEN_AI_API_KEY"
```

**Setting up the dummy DB for testing(Implement this before running the above command if you intend to test on the dummy database provided in the repo, below instructions aree for Google Colab and other clound environments):
**
```
sudo apt-get -y install mysql-server

sudo service mysql start

sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH 'mysql_native_password' BY 'root';FLUSH PRIVILEGES;"

mysql -u root -p (then input the password i.e. root)

mysql> source ~/database/mysqlsampledatabase.sql
```

## Example implementation on dummy database post running the code:

![alt text](image.png)

## Video Demo on Terminal:

https://github.com/avnishkanungo/AudioNLToSQL/assets/17869179/3b42f61d-e537-40b9-a2b7-4156c7e65d1c

## Demo Link: https://huggingface.co/spaces/avnishkanungo/AudioNLtoSQL

The demo for this project can be run in three ways, via the terminal command mentioned above, via the jupyter notebook(gradio_demo.ipynb) or by running the script app.py.

To run a demo of the code yourself, please install the required libraries using the command mentioned above, then follow one of the two options below:

- Using the jupyter notbeook: Using the gradio_demo.ipynb you can just run the cells make changes to the DB connection code, connect to your required DB type and run the web demo vis the gradio implementation that you can find in the botebook.
- Using the app.py script:  Similar to the notebook you can make the required changes to the DB connection setup to coneect to your database if needed and the files required for finetuing(database_table_descriptions.csv and few_shot_samples.json files) and run the app.py script to trigger the web app using gradio.
- Running the scipt on the terminal using the NLToSQL.py script: The instructions for the same can be found above.

NOTE: You will need to have the database that you want to connect to, active on your machine or hosted on a server, for the script to work without any issue. 

References: 
- https://blog.futuresmart.ai/mastering-natural-language-to-sql-with-langchain-nl2sql
- https://python.langchain.com/v0.1/docs/modules/model_io/prompts/few_shot_examples_chat/
- https://python.langchain.com/v0.1/docs/use_cases/sql/
- https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning
