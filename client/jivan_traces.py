import json
from dotenv import load_dotenv
load_dotenv(".env")

import mysql.connector

# Connection configuration
DB_NAME = "ragmetrics_dev_2"
DB_HOST = "127.0.0.1"
DB_PORT = "3307"
DB_USERNAME = "django_user"
DB_PASS = "password"
TABLE = "table2"

class StubLLMClient:
    def invoke(self, messages, output, **kwargs):
        return output

# Initialize RagMetrics with API key from environment.
import ragmetrics
ragmetrics.login(base_url="http://localhost:8000")

# Create and use the stub client.
stub_client = StubLLMClient()

def my_callback(raw_input, raw_output):
    # Your custom post-processing logic here. For example:
    input_msg = raw_input[-1]['content']
    output_str = raw_output['choices'][0]['message']['tool_calls'][0]['function']['arguments']
    output_data = json.loads(output_str)
    processed = {
         "input": input_msg,
         "output": output_data
    }
    return processed

# Monitor the stub client; RagMetrics will wrap the invoke() method.
ragmetrics.monitor(stub_client, context={"traces_version_3": "true"}, callback=my_callback)

# Instead of hardcoding the messages, read from the database.
try:
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USERNAME,
        password=DB_PASS,
        database=DB_NAME
    )
    cursor = conn.cursor()
    query = f"SELECT input, output FROM {TABLE}"
    cursor.execute(query)
    rows = cursor.fetchall()
    for idx, (input_val, output_val) in enumerate(rows, start=1):
        try:
            # Assume stored JSON strings; convert them into Python objects.
            messages = json.loads(input_val) if isinstance(input_val, str) else input_val
            out_data = json.loads(output_val) if isinstance(output_val, str) else output_val
        except Exception as parse_error:
            print(f"Error parsing database row {idx}: {parse_error}")
            continue

        response = stub_client.invoke(
            input=messages,
            output=out_data
        )
        print(f"Row {idx}: {response}")
    cursor.close()
    conn.close()
except Exception as db_error:
    print("Error reading from database:", db_error)
