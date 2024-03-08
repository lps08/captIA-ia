#%%
from src.pdf_extraction import extract_infos_gemini_google
from src.pdf_extraction import extract_infos_gemma
from src.pdf_extraction import extract_infos_manual
from src.ml_models.llm.retrieval_qa_llm import get_google_embeddings, get_huggingface_embeddings
from src.ml_models.llm.gemini_google_llm import get_gemini_model, get_parser
from src.ml_models.llm.gemma_llm import get_gemma_model
from src.ml_models.word2vec.similarity import load_model
from src.constants import ModelCard
from src import constants
import langchain
import os
langchain.debug = True #for debuging 

def get_pdf_infos(pdf_path, model_to_use:ModelCard = constants.MODEL_TO_USE):
    """
    Extracts information from a PDF file using different models based on the specified ModelCard.

    Args:
        pdf_path (str): The path to the PDF file.
        model_to_use (ModelCard): The model card to use for extraction.

    Returns:
        dict: A dictionary containing extracted information from the PDF.

    Note:
        This function is a higher-level interface that allows users to extract information from PDF files using different models based on the specified ModelCard. For more detailed information on the individual models and their capabilities, refer to the documentation of the respective modules:
        - For Gemini-Google model: `extract_infos_gemini_google.py`
        - For Gemma model: `extract_infos_gemma.py`
        - For manual extraction using Word2Vec: `extract_infos_manual.py`
    
    Example:
        >>> pdf_path = 'example.pdf'
        >>> model_to_use = ModelCard.GEMINI_GOOGLE
        >>> infos = get_pdf_infos(pdf_path, model_to_use)
        >>> print(infos)
    """
    if model_to_use == ModelCard.GEMINI_GOOGLE:
        gemini_llm = get_gemini_model()

        infos = extract_infos_gemini_google.extract_infos(
            pdf_path,
            llm=gemini_llm,
            embeddings=get_google_embeddings(),
            parser=get_parser()
        )
        return infos

    elif model_to_use == ModelCard.GEMMA:
        gemma_llm = get_gemma_model()
        
        infos = extract_infos_gemma.extract_infos(
            pdf_path,
            llm=gemma_llm,
            embeddings=get_huggingface_embeddings(),
        )
        return infos

    elif model_to_use == ModelCard.MANUAL:
        word2vec_model = load_model(os.path.join(constants.DATA_PATH, constants.WORD2VEC_MODEL_FILE))
        infos = extract_infos_manual.extract_infos(
            pdf_path, 
            model=word2vec_model,
        )
        return infos

if __name__ == '__main__':
    PDF_PATH = os.path.join(constants.EDITALS_DATASET_PATH, "cnpq.pdf")
    res = get_pdf_infos(PDF_PATH)
    print(res)