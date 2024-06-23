import os.path

import requests, chromadb, time
from chromadb import Documents, EmbeddingFunction, Embeddings, Client
from src.config import STN_TRANSFORMER, CHROMA_DB_DIR, TOP_K_ELEMENTS, RETRIES, SLEEP, EMBEDDING_MODEL, MODEL_DIR, logger
from typing import Tuple, List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class DistilbertLocalEmbedding(EmbeddingFunction):
    def __init__(self, mode: str = "local"):
        """
        Initializes the DistilbertLocalEmbedding class based on the model loading mode.

        :param mode: enable local model loading if the model is hosted locally. Online-mode otherwise
        """
        if mode not in ["local", "online"]:
            logger.error("Only local and online modes are available")
            raise Exception("Invalid embedding model loading mode")

        self.mode = mode
        self.embeddings = []
        if self.mode == "local":
            logger.info("Loading {} model in local...".format(EMBEDDING_MODEL))
            if MODEL_DIR:
                self.local_model = SentenceTransformer(model_name_or_path=os.path.join(MODEL_DIR, EMBEDDING_MODEL))
            else:
                self.local_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Model has been loaded successfully!")

    def __call__(self, input_text: Documents) -> Embeddings:
        """
        Generates embeddings for the input_text based on a local model inference or an API call.

        :param input_text: input document text for the embedding generation
        :return: embedded representation of the input document text based on the local model or the API call.
        """
        if self.mode == "local":
            a = self.local_model.encode(input_text).tolist()
            return a
        elif self.mode == "online":
            return self.__post(input_text, 0)

    def __post(self, input_text: Documents, retry_counter: int) -> List:
        """
        Sends input document to the sentence-transformers API to generate embeddings.

        :param input_text: input document text for the embedding generation
        :param retry_counter: number of times the API has been called for the current document.
        :return: embedded representation of the input document text
        """
        if retry_counter > RETRIES:
            logger.warning("Hugging Face service is not working properly, number of retries {}".format(retry_counter))
            return [None for d in input_text]

        try:
            response = requests.post(STN_TRANSFORMER["API_URL"],
                                     headers=STN_TRANSFORMER["HEADERS"],
                                     json={"inputs": input_text})

            if not response or response.status_code != 200:
                logger.warning("Response status code {}".format(response.status_code))
                time.sleep(SLEEP)
                self.__post(input_text, retry_counter + 1)
            else:
                return response.json()

        except Exception as e:
            logger.error("An exception has occurred during Embedding request\n", e)
        return [None for d in input_text]

class ChromaDB:
    def __init__(self, embedding_mode: str = "local"):
        self.db_path = CHROMA_DB_DIR
        self.top_k = TOP_K_ELEMENTS
        self.embedding_model = DistilbertLocalEmbedding(embedding_mode)

    def __call__(self, query: str, db_name: str, documents: List) -> List:
        """
        Creates or Loads the db instance to extract relevant passages based on the query.

        :param query: question of interest
        :param db_name: name of the db collection instance
        :param documents: list of all the identified chunks from the documents.
        :return: list of relevant passages based on the query.
        """
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.db = self.__create_or_load(db_name, documents)

        passages = self.__get_relevant_passages(query)
        return passages

    def __create_or_load(self, db_name: str, documents: List) -> Client():
        """
        Loads the db instance, if the name is found in the list of collections.
        Creates a new one, otherwise.

        :param db_name: name of the db collection instance.
        :param documents: list of all the identified chunks from the documents.
        :return: db collection instance
        """
        if db_name in [c.name for c in self.chroma_client.list_collections()]:
            logger.info("DB already exists, Loading...")
            return self.__load_chroma_db(db_name, documents)
        else:
            return self.__create_chroma_db(db_name, documents)

    def __load_chroma_db(self, db_name: str, documents: List):
        """
        Loads the db instance based on the db_name and aggregates the documents that are not part of the collection.

        :param db_name: name of the db collection instance.
        :param documents:list of all the identified chunks from the documents.
        :return:db collection instance
        """
        db = self.chroma_client.get_collection(name=db_name, embedding_function=self.embedding_model)
        all_metadata = db.get(include=["metadatas"]).get('metadatas')
        distinct_files = set([x.get('file_name') for x in all_metadata])
        max_id = db.count() + 1

        count_new_files = 0
        for i, d in enumerate(tqdm(documents)):
            if d[1] not in distinct_files:
                db.add(documents=[d[0]], ids=[str(max_id + i)], metadatas=[{"file_name": d[1]}])
                count_new_files += 1
        logger.info("New chunks added: {}".format(count_new_files))
        return db


    def __create_chroma_db(self, db_name: str, documents: List) -> Client():
        """
        Creates the db instance based on the db_name and aggregates all the documents of the list.

        :param db_name: name of the db collection instance.
        :param documents: list of all the identified chunks from the documents.
        :return: db collection instance
        """
        db = self.chroma_client.create_collection(name=db_name, embedding_function=self.embedding_model)

        logger.info("Embedding creation process has started...")
        for i, d in enumerate(tqdm(documents)):
            db.add(documents=[d[0]], ids=[str(i)], metadatas=[{"file_name": d[1]}])
        return db

    def __get_relevant_passages(self, query: str) -> List:
        """
        Extracts the top-k most relevant passages from the collection based on the input query.
        The query is performed using the cosine similarity.

        :param query: question of interest
        :return: a list of the top_k most relevant passages for the input query.
        """
        passages = self.db.query(query_texts=[query], n_results=self.top_k, include=['distances', 'documents'])
        passages = [{"id": str(i), "snippet": str(text)} for i, text in zip(passages.get("ids")[0], passages.get("documents")[0])]
        return passages

    def return_passage(self, id: int) -> Tuple[str, str]:
        """
        Retrieves the text and the name of the document based on the id.

        :param id: identification number of the passage needed.
        :return: the text of the passage and the name of the document from where it was extracted.
        """
        register = self.db.get(ids=[str(id)], include=['documents', 'metadatas'])
        passage = register.get("documents")[0] if register.get("documents") else "Passage not found"
        document_name = register.get("metadatas")[0].get("file_name", "File not found") if register.get("metadatas") else "File not found"
        return passage, document_name
