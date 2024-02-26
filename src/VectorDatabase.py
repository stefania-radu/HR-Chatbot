import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
import chromadb

logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(self, docs_by_ids={}, embeddings_function=OpenAIEmbeddings, config=None):
        """
        Initialize the VectorDatabase class.

        Args:
        - docs_by_ids (dict): A dictionary containing document IDs as keys and document contents as values.
        - embeddings_function (class): The class representing the embeddings function to be used.
        - config (object): The configuration object.
        """
        self.directory_name = config.data_directory
        self.embeddings_function = embeddings_function
        self.persist_directory = config.persist_directory
        self.docs_by_ids = docs_by_ids
        self.collection_name = "langchain_store"

        self._create_embeddings()
        self._create_vector_database()


    def _create_embeddings(self):
        """
        Create the embeddings using the specified embeddings function.
        """
        self.embeddings = self.embeddings_function()
        print(f"Embeddings created using {self.embeddings_function.__name__}.")


    def _create_vector_database(self):
        """
        Create the vector database using the specified documents and embeddings.
        """
        ids = list(self.docs_by_ids.keys())
        documents = list(self.docs_by_ids.values())

        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory, settings=Settings(anonymized_telemetry=False))
        self.vector_database = Chroma(collection_name=self.collection_name, persist_directory=self.persist_directory, embedding_function=self.embeddings, client=self.chroma_client)
        self.vector_database.add_documents(documents=documents, ids=ids, embeddings=self.embeddings)
        print(f"Vector database created in {self.persist_directory}.")
        self.vector_database.persist()


    def update_database(self, docs_by_ids):
        """
        Update the vector database with new documents.

        Args:
        - docs_by_ids (dict): A dictionary containing document IDs as keys and document contents as values.

        Returns:
        None
        """
        print(f"Appending to existing vectorstore at {self.persist_directory}")

        ids = list(docs_by_ids.keys())
        documents = list(docs_by_ids.values())

        self.vector_database.add_documents(documents=documents, ids=ids)

        self.vector_database.persist()


    def get_database_as_retriever(self, k=5):
        """
        Get the vector database as a retriever.

        Args:
        - k (int): The number of nearest neighbors to retrieve.

        Returns:
        The vector database as a retriever.
        """
        return self.vector_database.as_retriever(search_kwargs={"k": k})
