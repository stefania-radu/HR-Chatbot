import os

class Config:
    def __init__(self):
        self.project_name = "chatbot"
        self.session_id = "abc123"
        self.data_directory = "data"
        self.ignored_files = []
        self.persist_directory = "chroma_db"
        self.model_first_instruction = "You are a helpful robot who engages in conversations with the users and answers questions."

    def __str__(self):
        return f"Config(project_name={self.project_name}, \n \
            session_id={self.session_id}, \n \
            data_directory={self.data_directory}, \n \
            ignored_files={self.ignored_files}, \n \
            persist_directory={self.persist_directory}, \n \
            model_first_instruction={self.model_first_instruction}, \n"