import os
from dotenv import load_dotenv
import logging, sys

load_dotenv()

EMBEDDING_MODEL = "multi-qa-distilbert-cos-v1"

STN_TRANSFORMER = {
    "API_URL": "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/{}".format(EMBEDDING_MODEL),
    "HEADERS": {"Authorization": "Bearer {}".format(os.getenv("HUGGING_FACE_TOKEN"))}
}

COHERE_MODEL = {
    "API_URL": "",
    "TOKEN": os.getenv("COHERE_TOKEN")
}

CHROMA_DB_DIR = os.getenv("CHROMA_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")

# VARIABLES
TOP_K_ELEMENTS = 3
MIN_WORDS = 25
MAX_WORDS = 1000
RETRIES = 3
SLEEP = 5

# EXTENSIONS
WORD = ['doc', 'docx']
PPT = 'pptx'
EMAIL = 'msg'
PDF = 'pdf'
EXCEL = ['xlsx', 'csv', 'xls']

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('RAG System')
logger.setLevel(logging.INFO)





