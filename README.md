# HR Assistant using LangChain and OpenAI

![intro](https://github.com/stefania-radu/HR-Chatbot/assets/82370258/9e0f7ad7-af49-4c98-a41a-fb9d1bb08766)

## Description
This project utilizes a LangChain 🦜🔗 agent based on the OpenAI  LLM, capable of reasoning using a set of tools, including internet search and Retrieval-Augmented Generation (RAG) using Chroma DB. The agent's reasoning ability allows it to dynamically decide the most appropriate actions and answers in various contexts. 

The agent can answer questions based on .pdf and .txt documents. ChatGPT was used to generate examples for an HR contract and a company policy. These examples can be found in the [data](data) folder. To retrieve relevant information, I used a Chroma-based retriever and OpenAI embeddings.

## Installation

Install the requirements:
- `pip install -r requirements`

API keys are required from:
- OpenAI: for the LLM
- Tavily: for the web search tool
- LangSmith: for monitoring the chain

## Demo

### [Terminal](main.py)

Run `python main.py` to run chat with the agent using the terminal.


### [Jupyter Notebook](demo.ipynb)

A step by step walkthrough of the project, from loading the data and creating the vector database, to using the model to chat with your data.

### [Flask App](app.py)

Run `python app.py` to create a small app that runs in your local web server. It uses Flask for the web interface and multiprocessing to handle chatbot operations.

*Note: The memory is not supported yet in the app version.*

![benfits](https://github.com/stefania-radu/HR-Chatbot/assets/82370258/28d10f4e-0b92-4635-8b68-9011e88cd6b9)
![insurance](https://github.com/stefania-radu/HR-Chatbot/assets/82370258/f1a58160-1732-41a0-941a-1c5e10babe25)
