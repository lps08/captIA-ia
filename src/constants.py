from configparser import ConfigParser
from dotenv import load_dotenv
import enum

load_dotenv()

class ModelCard(enum.Enum):
    GEMINI_GOOGLE = "Google gemini online LLM model"
    GEMMA = "Gemma local LLM model"
    MANUAL = "Manual approach"

CAPTIA_CONFIG_PATH = "config/app_config.ini"
config = ConfigParser()
config.read(CAPTIA_CONFIG_PATH)

CONFIG_PATH = config.get('captia', 'config_path')
DATA_PATH = config.get('captia', 'data_path')
SITES_CONFIG_FILE = config.get('captia', 'sites_config_file')
BERT_MODEL_NAME = config.get('captia', 'bert_model_name')
BERT_FINETUNED_MODEL_FOLDER = config.get('captia', 'bert_finetuned_model_folder')
MODEL_TO_USE = ModelCard.GEMINI_GOOGLE
WORD2VEC_MODEL_FILE = config.get('captia', 'word2vec_model_file')
QA_MODEL_NAME = config.get('captia', 'qa_model_name')
URL_WOR2VEC_MODEL = config.get('captia', 'url_word2vec_model')
EDITALS_DATASET_PATH = config.get('captia', 'editals_dataset_path')
GOOGLE_EMBEDDINGS_MODEL_NAME = config.get('captia', 'google_embeddings_model_name')
HUGGINGFACE_EMBEDDINGS_MODEL_NAME = config.get('captia', 'huggingface_embeddings_model_name')
GEMMA_MODEL_NAME = config.get('captia', 'gemma_model_name')
GOOGLE_GEMINI_MODEL_NAME = config.get('captia', 'google_gemini_model_name')