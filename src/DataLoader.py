import os
import glob
import logging

logger = logging.getLogger(__name__)

from multiprocessing import Pool
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)


DOCS_MAP = {
    ".pdf": (PyPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"})
}


class DataLoader:
    def __init__(self, config=None):
        self.directory_name = config.data_directory
        self.ignored_files = config.ignored_files


    def _load_single_document(self, file_path):
        ext = "." + file_path.rsplit(".", 1)[-1].lower()
        if ext in DOCS_MAP:
            loader_class, loader_args = DOCS_MAP[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()

        print(f"Unsupported file extension '{ext}'")


    def _load_documents(self, directory_name, ignored_files):
        """
        Loads all documents from the source documents directory, ignoring specified files
        """
        all_files = []
        for ext in DOCS_MAP:
            all_files.extend(
                glob.glob(os.path.join(directory_name, f"**/*{ext.lower()}"), recursive=True)
            )
            all_files.extend(
                glob.glob(os.path.join(directory_name, f"**/*{ext.upper()}"), recursive=True)
            )
            
        filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

        print(f"Loading documents from {self.directory_name}")
        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
                for i, docs in enumerate(pool.imap_unordered(self._load_single_document, filtered_files)):
                    for doc in docs:
                        if doc not in results:
                            results.append(doc)
                            pbar.update()
        print(results)
        print(f"Done loading {len(results)} documents from {self.directory_name}")
        return results


    def process_documents(self, ignored_files=[], chunk_size=500, chunk_overlap=0):
        """
        Load documents and split in chunks
        """
        print(f"Processing documents from {self.directory_name}")

        documents = self._load_documents(self.directory_name, ignored_files)
        if not documents:
            print("No new documents to load")
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.processed_documents = text_splitter.split_documents(documents)
        
        print(f"Split documents into {len(self.processed_documents)} chunks of text (max. {chunk_size} tokens each).")

        ids = self._generate_ids(self.processed_documents)
        
        return dict(zip(ids, self.processed_documents))


    def _generate_ids(self, items_list):
        ids = [str(i) for i in range(len(items_list))]
        return set(ids)