import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

from config.Config import Config
from src.VectorDatabase import VectorDatabase
from src.DataLoader import DataLoader
from src.AgentChat import AgentChat
from langchain.callbacks.tracers import LangChainTracer


config = Config()
print(f"Project *{config.project_name}* initialized from configuration: \n {str(config)} \n")

# keep track of API calls
tracer = LangChainTracer(project_name=config.project_name)


def main():

    dataloader = DataLoader(config=config)
    docs_by_ids = dataloader.process_documents()
    
    vdb = VectorDatabase(docs_by_ids=docs_by_ids, config=config)
    retriever = vdb.get_database_as_retriever()

    chatbot = AgentChat(retriever=retriever)

    while True:
        user_input = input("Enter your message: ")
        if user_input == "exit":
            exit()
        elif user_input.strip() == '':
            continue
        response = chatbot.get_response(user_input, config_invoke={"configurable": {"session_id": config.session_id}, "callbacks": [tracer]})
        print(f"response: {response['output']}")

    

if __name__ == "__main__":
    main()
    
