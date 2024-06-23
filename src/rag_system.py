from src.preprocess.preprocessing import load_files
from src.vector_databse.build_database import ChromaDB
from src.llm.LLM import LLM
from src.config import logger
import os


class RAGSystem:
    def __init__(self, mode="local"):
        self.vector_db = ChromaDB(embedding_mode=mode)
        self.llm = LLM()
        self.data_dir = {}

    def __call__(self, folder_dir: str, query: str):
        """
        Creates a RAG system based on the data extracted from the files in the folder_directory.
        Identifies the most relevant passage and document based on que input query and
        the vector database from the RAG system.

        :param folder_dir: directory of the data files to load in the RAG system.
        :param query: question of interest
        """
        logger.info("Starting RAG system...")
        db_name = os.path.basename(folder_dir).split(".")[0]
        num_files = len(os.listdir(folder_dir))

        if db_name not in self.data_dir or self.data_dir[db_name] != num_files:
            logger.info("Loading files and extracting text...")
            documents = load_files(folder_dir)
        else:
            logger.info("Directory already exists and no more files have been added")
            documents = []

        passages = self.vector_db(query, db_name, documents)
        logger.info("Sending query to llm...")
        answer, doc_reference = self.llm(query, passages)
        logger.info("ANSWER: {}".format(answer))

        if doc_reference != -1:
            passage, document = self.vector_db.return_passage(doc_reference)
            logger.info("Extracted from the document {}".format(document))
            logger.info("---------------------------- Passage below ----------------------------\n")
            logger.info(passage)





