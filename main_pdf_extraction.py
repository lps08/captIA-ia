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
from src.database.db import EditalDatabse, ScrapingDatabase
import requests
import tempfile
from dateparser.search import search_dates
import re

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

def parse_areas(areas_list, max_size=3):
    """
    Parses and retrieves a subset of areas of knowledge from a list of areas.

    This function sorts the list of areas of knowledge based on their lengths, removes empty strings, and retrieves a 
    subset of the areas with the shortest lengths, up to a maximum size specified by the `max_size` parameter.

    Args:
        areas_list (list of str): The list of areas of knowledge.
        max_size (int, optional): The maximum number of areas to retrieve. Defaults to 3.

    Returns:
        list of str: A subset of areas of knowledge, sorted by length, with empty strings removed, and limited to `max_size`.

    Example:
        To parse and retrieve a subset of areas of knowledge from a list of areas, you can use this function as follows:

        >>> areas_list = ['Engineering', 'Computer Science', 'Mathematics', 'Physics']
        >>> parsed_areas = parse_areas(areas_list)
        >>> print(parsed_areas)
        ['Engineering', 'Physics', 'Mathematics']

    """
    areas_list = [i.strip() for i in areas_list if i != '']
    areas_list.sort(key=len)
    return areas_list[:max_size]

def parse_money_value(text):
    """
    Parses and extracts a monetary value from a given text.

    This function uses a regular expression to search for and extract monetary 
    values from the input text. It supports various currency symbols, including 
    pound (£), dollar ($), euro (€), and Brazilian real (R$), as well as different 
    formats for expressing numbers, such as using commas or periods as decimal separators.

    Args:
        text (str): The text from which to extract the monetary value.

    Returns:
        str: The extracted monetary value from the text.

    Example:
        To parse and extract a monetary value from a text, you can use this function as follows:

        >>> text = "The price of the product is $25.99."
        >>> money_value = parse_money_value(text)
        >>> print(money_value)
        '$25.99'
    """
    money_regex = re.compile(r"([£\$€]|(R\$))\s*(\d+(?:[\.\,]\d+)*)\s*(milhões|milhão|mil)?|(\d+(?:[\.\,]\d+)*)\s*([£\$€]|(mil)?\s*(euros|R\$|reais|milhão|milhões|mil))")
    res = money_regex.search(text)
    return res.group() if res else text

def extract_pdf_infos_db(model_to_use:ModelCard = constants.MODEL_TO_USE):
    """
    Extracts information from PDF files stored in the database and saves them into another database.

    Parameters:
        model_to_use (ModelCard): The model to use for extracting PDF information.

    Returns:
        None

    Notes:
        This function retrieves PDF files from a database table, extracts information from them using
        the specified model, and saves the extracted information into another database table.

        It fetches PDF files from the 'link_pdf' column of the 'scraping_table_name' table and iterates
        over each entry. For each PDF, it downloads the file, extracts information using the specified
        model, and then inserts the extracted information into the 'editals_table_name' table.

        If any error occurs during the process, it prints an error message and continues to the next PDF.

    Example:
        # Assuming constants.MODEL_TO_USE is the desired model to use
        # and constants.DATA_PATH, constants.SQLITE_DB_FILE, constants.SCRAPING_TABLE_NAME,
        # constants.EDITALS_TABLE_NAME are defined correctly.
        extract_pdf_infos_db(constants.MODEL_TO_USE)
    """
    db_path = os.path.join(constants.DATA_PATH, constants.SQLITE_DB_FILE)
    scraping_db = ScrapingDatabase(db_path, constants.SCRAPING_TABLE_NAME)
    editals_db = EditalDatabse(db_path, constants.EDITALS_TABLE_NAME)
    
    editals_saved = scraping_db.get_all()
    if len(editals_saved) > 0:
        for edital in editals_saved:
            try:
                response = requests.get(edital['ds_link_pdf'])
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf_file:
                    temp_pdf_file.write(response.content)
                    temp_pdf_path = temp_pdf_file.name
                    infos = get_pdf_infos(temp_pdf_path, model_to_use)

                    editals_db.insert_data(
                        ds_link_pdf=edital['ds_link_pdf'],
                        ds_agency=edital['ds_agency'].upper(),
                        ds_titulo=infos['titulo'],
                        ds_objetivo=infos['objetivo'],
                        ds_elegibilidade='; '.join(infos['elegibilidade']) if type(infos['elegibilidade']) == list else infos['elegibilidade'],
                        dt_submissao=search_dates(infos['submissao'])[-1][1] if search_dates(infos['submissao']) else infos['submissao'],
                        ds_financiamento=parse_money_value(infos['financiamento']),
                        ds_areas='; '.join(parse_areas(infos['areas'])) if type(infos['areas']) == list else infos['areas'],
                        ds_nivel_trl=infos['nivel_trl'],
                    )

            except Exception as e:
                print(f"Error on pdf {edital['ds_link_pdf']} -> {e}")
                pass
    else:
        raise Exception(f"No pdfs found in {scraping_db.table_name} table!")

    scraping_db.close()
    editals_db.close()
    
if __name__ == '__main__':
    # PDF_PATH = os.path.join(constants.EDITALS_DATASET_PATH, "cnpq.pdf")
    # res = get_pdf_infos(PDF_PATH)
    # print(res)
    extract_pdf_infos_db()