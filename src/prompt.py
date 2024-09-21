

import os
from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()

def generate_answer(query, qa_chain):
    return qa_chain.run(query)
