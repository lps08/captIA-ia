#%%
from src.pdf_extraction import extract_infos_gemini_google
from src.pdf_extraction import extract_infos_local_llm_model
from src.pdf_extraction import extract_infos_manual
from src.ml_models.llm.retrieval_qa_llm import get_google_embeddings, get_huggingface_embeddings
from src.ml_models.llm.gemini_google_llm import get_gemini_model, get_parser
from src.ml_models.llm.local_llm_model import get_local_llm_model
from src.ml_models.word2vec.similarity import load_model
from src.constants import ModelCard
from src.scraping.pdf_scraping import PDFScraping
from src import constants
import langchain
import os
from src.database.db import EditalDatabse, ScrapingDatabase
import requests
import tempfile
from dateparser.search import search_dates
import re
from tqdm.notebook import tqdm
from configparser import ConfigParser

langchain.debug = True #for debuging 

def get_pdf_infos(edital_path, is_document_pdf, use_attachment_files=False, list_edital_attachment=[], model_to_use:ModelCard = constants.MODEL_TO_USE):
    """
    Extracts information from a PDF document using the specified model.

    This function determines which model to use for extracting information from the PDF document based on the `model_to_use` parameter. It supports three models: GEMINI_GOOGLE, LOCAL_LLM, and MANUAL. Each model uses a different approach for information extraction.

    Args:
        edital_path (str): The path to the PDF document.
        is_document_pdf (bool): Indicates whether the document is a PDF.
        use_attachment_files (bool, optional): Whether to use attachment files associated with the document. Defaults to False.
        list_edital_attachment (list, optional): A list of attachment file paths. Defaults to an empty list.
        model_to_use (ModelCard, optional): The model to use for extracting information. Defaults to constants.MODEL_TO_USE.

    Returns:
        dict: The extracted information as a dictionary.

    Example:
        >>> edital_path = "/path/to/document.pdf"
        >>> is_document_pdf = True
        >>> infos = get_pdf_infos(edital_path, is_document_pdf, use_attachment_files=True, list_edital_attachment=["/path/to/attachment1.pdf"], model_to_use=ModelCard.GEMINI_GOOGLE)
        >>> print(infos)
    """
    if model_to_use == ModelCard.GEMINI_GOOGLE:
        gemini_llm = get_gemini_model()

        infos = extract_infos_gemini_google.extract_infos(
            edital_path,
            llm=gemini_llm,
            embeddings=get_google_embeddings(),
            is_document_pdf=is_document_pdf,
            use_attachment_files=use_attachment_files,
            list_edital_attachment=list_edital_attachment,
            use_unstructured=True,
        )
        return infos

    elif model_to_use == ModelCard.LOCAL_LLM:
        local_llm_model = get_local_llm_model()
        
        infos = extract_infos_local_llm_model.extract_infos(
            edital_path,
            llm=local_llm_model,
            embeddings=get_huggingface_embeddings(),
            is_document_pdf=is_document_pdf,
            use_attachment_files=use_attachment_files,
            list_edital_attachment=list_edital_attachment,
            use_unstructured=True,
        )
        return infos

    elif model_to_use == ModelCard.MANUAL:
        word2vec_model = load_model(os.path.join(constants.DATA_PATH, constants.WORD2VEC_MODEL_FILE))
        infos = extract_infos_manual.extract_infos(
            edital_path, 
            model=word2vec_model,
        )
        return infos

def remove_unknown_characters(text, unknown_characters_dict = {"\x00": "", "�": "ti"}):
    """
    Removes unknown characters from text.

    This function takes a text string and removes any unknown characters based on the specified dictionary of unknown characters and their replacements.

    Args:
        text (str): The text string containing unknown characters.
        unknown_characters_dict (dict): A dictionary containing unknown characters as keys and their corresponding replacements as values.

    Returns:
        str: The text string with unknown characters removed.

    Notes:
        This function iterates through the dictionary of unknown characters and their replacements, replacing each occurrence of an unknown character in the input text with its corresponding replacement.

    Example:
        To remove unknown characters from a text string, you can use this function as follows:

        >>> text = "This is a test string with unknown characters � and \x00."
        >>> cleaned_text = remove_unknown_characters(text)
        >>> print(cleaned_text)
        'This is a test string with unknown characters ti and .'
    """
    for unk_char in unknown_characters_dict:
        text = text.replace(unk_char, unknown_characters_dict[unk_char])
    return text

def remove_a_lot_of_spaces_from_text(text):
    """
    Removes excessive spaces and newline characters from text.

    This function removes excessive spaces and newline characters from the input text using regular expressions.

    Args:
        text (str): The input text containing excessive spaces and newline characters.

    Returns:
        str: The text with excessive spaces and newline characters removed.

    Notes:
        This function uses regular expressions to replace consecutive whitespace characters (including spaces and newline characters) with a single space character.

    Example:
        To remove excessive spaces and newline characters from a text, you can use this function as follows:

        >>> text = "This   is     a    text    with\n excessive     spaces     and\n\nnewline characters."
        >>> cleaned_text = remove_a_lot_of_spaces_from_text(text)
        >>> print(cleaned_text)
        'This is a text with excessive spaces and newline characters.'
    """
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_list(text_list):
    """
    Parses a list of text items into a single string.

    This function takes a list of text items and parses them into a single string by joining them with a semicolon (;) delimiter. It also removes excessive spaces and newline characters from the resulting string.

    Args:
        text_list (list or str): The list of text items to be parsed. If a string is provided, it is assumed to be a semicolon-separated list.

    Returns:
        str: The parsed text string.

    Notes:
        This function first joins the text items in the list using a semicolon (;) delimiter. It then removes excessive spaces and newline characters from the resulting string using the remove_a_lot_of_spaces_from_text function.

    Example:
        To parse a list of text items into a single string, you can use this function as follows:

        >>> text_list = ["Item 1", " Item 2 ", "Item   3", " Item 4  "]
        >>> parsed_text = parse_list(text_list)
        >>> print(parsed_text)
        'Item 1;Item 2;Item 3;Item 4'
    """
    text = ";".join(text_list) if type(text_list) == list else text_list
    text = re.sub(r'(\s*;+\s*)+', ';', text)
    text = remove_a_lot_of_spaces_from_text(text)
    text = remove_unknown_characters(text)
    return text

def parse_areas(areas_list, max_size=3, default_value="Nao encontrado"):
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
    if len(areas_list) > 1:
        areas_list = [i.strip() for i in areas_list if i != '']
        areas_list.sort(key=len)
        areas_list_top = areas_list[:max_size]
        text_areas = parse_list(areas_list_top)
    else:
        text_areas = default_value
    return text_areas

def parse_elegibilidade(text):
    """
    Parses eligibility criteria from text.

    This function takes a text string containing eligibility criteria and parses it into a formatted string. It first removes excessive spaces and newline characters from the input text using the parse_list function.

    Args:
        text (str): The text string containing eligibility criteria.

    Returns:
        str: The parsed eligibility criteria string.

    Notes:
        This function calls the parse_list function to remove excessive spaces and newline characters from the input text. The resulting string is returned as the parsed eligibility criteria.

    Example:
        To parse eligibility criteria from a text string, you can use this function as follows:

        >>> text = ["Eligibility criteria 1", "Eligibility criteria 2", "Eligibility criteria 3"]
        >>> parsed_elegibilidade = parse_elegibilidade(text)
        >>> print(parsed_elegibilidade)
        'Eligibility criteria 1;Eligibility criteria 2;Eligibility criteria 3'
    """
    text = parse_list(text)
    text += "." if text[-1] != "." else ""
    return text

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

def parse_currency(text):
    """
    Parses and returns the currency symbol from a given text.

    This function identifies common currency symbols (e.g., R$, £, $, €) within the input text. 
    If no standard currency symbol is found, it attempts to match common currency names like 
    'real', 'euro', and 'dólar' (or their variations) and returns the corresponding symbol.

    Args:
        text (str): The input text from which to parse the currency symbol.

    Returns:
        str: The matched currency symbol ('R$', '£', '$', '€'), or None if no currency is found.

    Examples:
        >>> parse_currency("O valor é R$ 1000.")
        'R$'
        
        >>> parse_currency("The price is $50.")
        '$'
        
        >>> parse_currency("Preço: 100 euros.")
        '€'
        
        >>> parse_currency("Custo em reais.")
        'R$'
        
        >>> parse_currency("Monto: 2000 dólares")
        '$'
        
        >>> parse_currency("No currency here.")
        None
    """
    currency_pattern = re.compile(r"(R?[£\$€])", re.IGNORECASE)

    currency_matched = currency_pattern.search(text)

    if not currency_matched:
        if re.search(r'rea(l|is)', text):
            return "R$"
        elif re.search(r'(euro)s?', text):
            return "€"
        elif re.search(r'd[o|ó]lar(es)?', text):
            return "$"
        elif re.search(r'(libra)s?', text):
            return "£"
        else:
            return None
    else:
        return currency_matched.group()

def parse_financing_value(text):
    """
    Parses and returns the financing value from a given text.

    This function identifies and extracts numerical values representing monetary amounts from the input text.
    It handles both formatted numbers (e.g., "1.234,56") and plain numbers (e.g., "1234" or "1,234").
    It also recognizes and converts values expressed in words (e.g., "milhão", "bilhão").

    Args:
        text (str): The input text from which to parse the financing value.

    Returns:
        float: The parsed financing value, or None if no value is found.

    Examples:
        >>> parse_financing_value("O valor do financiamento é 1.234.567,89.")
        1234567.89

        >>> parse_financing_value("O valor do financiamento é 2 bilhões.")
        2000000000.0

        >>> parse_financing_value("O valor do financiamento é 5 mil.")
        5000.0

        >>> parse_financing_value("No financing value here.")
        None
    """
    financing_pattern = re.compile(r"(\d{1,3}\.)+(\d{1,3})(,\d{1,2})?")
    financing_matched = financing_pattern.search(text)
    financing_value = financing_matched.group().replace('.', '').replace(',', '.') if financing_matched else None

    if financing_value is None:
        num_text = re.search(r"\d{1,3}(,\d{1})?", text)
        
        if num_text:
            num_text = num_text.group()
            num_value = float(num_text.replace(',', '.'))

            if re.search(r'bilh(ao|ão|oes|ões)', text):
                return num_value * 1000000000
            elif re.search(r'milh(ao|ão|oes|ões)', text):
                return num_value * 1000000
            elif re.search(r'mil', text):
                return num_value * 1000
            else:
                return None
        
        else:
            return None

    else:
        return float(financing_value)

def parse_edital_number(text):
    """
    Parses and extracts an edital number from a given text.

    This function uses a regular expression to search for and extract edital numbers from the input text. 
    It identifies edital numbers as sequences of digits separated by a forward slash (/), typically in 
    the format "### / ####".

    Args:
        text (str): The text from which to extract the edital number.

    Returns:
        str: The extracted edital number from the text, or 'Não encontrado' if no edital number is found.

    Notes:
        The regular expression used in this function assumes that the edital number follows the format 
        "### / ####". It may not capture all possible variations of edital number formats, and may require 
        adjustments for specific cases.

    Example:
        To parse and extract an edital number from a text, you can use this function as follows:

        >>> text = "O edital número 123/2022 foi lançado recentemente."
        >>> edital_number = parse_edital_number(text)
        >>> print(edital_number)
        '123/2022'
    """
    number_regex = re.compile(r"\b(?!(?:\d+\/\d+\/\d+))(\d{1,3}\s?\/\s?\d{4})\b")
    res = number_regex.search(text)
    return res.group() if res else 'Não encontrado'

def parse_title(list_titles, min_lenght=18, min_words_merge=5):
    """
    Parses and extracts a title from a list of titles.

    This function processes a list of titles to extract a single, coherent title based on certain criteria. It merges short titles or titles containing specific keywords into a longer title, and removes unnecessary characters.

    Args:
        list_titles (list): A list of titles to process.
        min_lenght (int): The minimum length required for a title to be considered valid. Defaults to 18.
        min_words_merge (int): The minimum number of words required in a title to prevent merging with other titles. Defaults to 5.

    Returns:
        str: The parsed and extracted title.

    Notes:
        This function aims to extract a single, comprehensive title from a list of titles by merging short titles or titles containing specific keywords into a longer title. It may not capture all variations of title formats, and may require adjustments for specific cases.

    Example:
        To parse and extract a title from a list of titles, you can use this function as follows:

        >>> list_of_titles = ["Chamada para propostas", "Edital de seleção de projetos"]
        >>> title = parse_title(list_of_titles)
        >>> print(title)
        'Edital de seleção de projetos'
    """
    chamada_edital_regex = re.compile(r'\b(edital|chamada)\b', re.IGNORECASE)
    title = list_titles[-1]

    if len(title) <= min_lenght or (chamada_edital_regex.search(title) and len(title.split()) <= min_words_merge):
        title = ' '.join(list_titles[-2:])

    title_splitted = title.split('–')
    title_splitted = [t.strip() for t in title_splitted]

    if len(title_splitted) > 1:
        range_list_accepted = -1
        for i in range(1,len(title_splitted)):
            new_title = ' '.join(title_splitted[range_list_accepted:]).strip()
            if len(new_title) <= min_lenght:
                range_list_accepted -= 1
        title = new_title if len(new_title) >= min_lenght else title
    title = remove_a_lot_of_spaces_from_text(title)
    title = remove_unknown_characters(title)
    return title

def parse_full_title(text):
    """
    Parses the full title text.

    This function takes a full title text string and performs preprocessing by removing unknown characters and extra spaces.

    Args:
        text (str): The full title text string to be parsed.

    Returns:
        str: The parsed full title text string.

    Notes:
        This function first removes any unknown characters from the input text using the `remove_unknown_characters` function. Then, it removes extra spaces using the `remove_a_lot_of_spaces_from_text` function.

    Example:
        To parse a full title text string, you can use this function as follows:

        >>> full_title_text = "This is a full title text with unknown \n\ncharacters � and extra    spaces."
        >>> parsed_text = parse_full_title(full_title_text)
        >>> print(parsed_text)
        'This is a full title text with unknown characters and extra spaces.'
    """
    text = remove_unknown_characters(text)
    text = remove_a_lot_of_spaces_from_text(text)
    return text

def parse_objetivo(text):
    """
    Parses the objective text.

    This function takes an objective text string and performs preprocessing by removing unknown characters, extra spaces, and ensuring proper capitalization and punctuation.

    Args:
        text (str): The objective text string to be parsed.

    Returns:
        str: The parsed objective text string.

    Notes:
        This function first removes any unknown characters from the input text using the `remove_unknown_characters` function. Then, it removes extra spaces using the `remove_a_lot_of_spaces_from_text` function. After that, it ensures that the first letter of the text is capitalized and adds a period at the end if it's missing.

    Example:
        To parse an objective text string, you can use this function as follows:

        >>> objective_text = "this is an objective text with unknown characters � and extra spaces"
        >>> parsed_text = parse_objetivo(objective_text)
        >>> print(parsed_text)
        'This is an objective text with unknown characters and extra spaces.'
    """
    text = remove_unknown_characters(text)
    text = remove_a_lot_of_spaces_from_text(text)
    text = text[0].upper() + text[1:]
    text += "." if text[-1] != "." else ""
    return text

def parse_datetime(text, defaul_return="Fluxo contínuo", language='br', date_order='DMY', day_of_month='first'):
    """
    Parses datetime information from text.

    This function parses datetime information from text using the `search_dates` function from the `dateparser` library. It searches for datetime expressions in the specified language and returns the parsed date.

    Args:
        text (str): The text containing datetime information.
        defaul_return (str): The default value to return if no date is found. Defaults to "Fluxo contínuo".
        language (str): The language of the datetime expressions in the text. Defaults to 'br' (Brazilian Portuguese).
        date_order (str): The order of date components (day, month, year) in the datetime expressions. Defaults to 'DMY' (day, month, year).
        day_of_month (str): The preference for day of the month ('first' or 'last') in ambiguous cases. Defaults to 'first'.

    Returns:
        str: The parsed datetime information, or the default value if no date is found.

    Notes:
        This function utilizes the `search_dates` function from the `dateparser` library to parse datetime expressions from text. It allows for flexible parsing of datetime information in various languages and formats.

    Example:
        To parse datetime information from text, you can use this function as follows:

        >>> text = "A data limite de submissão é 30 de junho de 2022."
        >>> parsed_date = parse_datetime(text)
        >>> print(parsed_date)
        '2022-06-30 00:00:00'
    """
    date_found = search_dates(text, languages=[language], settings={'DATE_ORDER': date_order, 'PREFER_DAY_OF_MONTH': day_of_month})
    date = date_found[-1][1] if date_found else defaul_return
    return date

def parse_nivel_trl(text):
    """
    Parses the Technology Readiness Level (TRL) from text.

    This function extracts the Technology Readiness Level (TRL) information from text using regular expressions. It searches for TRL values in various formats and returns the extracted TRL level.

    Args:
        text (str): The text containing TRL information.

    Returns:
        str: The extracted TRL level, or 'Não encontrado' (Not found) if no TRL level is found.

    Notes:
        This function uses regular expressions to search for TRL values in text. It searches for TRL levels expressed as single numbers or ranges (e.g., "3", "7 ou 8", "4 a 6") and returns the extracted TRL level. If no TRL level is found in the text, it returns 'Não encontrado'.

    Example:
        To extract the Technology Readiness Level (TRL) from text, you can use this function as follows:

        >>> text = "O nível de maturidade tecnológica necessário é 5 a 7."
        >>> trl_level = parse_nivel_trl(text)
        >>> print(trl_level)
        '5 a 7'
    """
    nivel_trl_regex = re.compile(r"\b([0-9],\s*)?[0-9](?:\s*(?:a|e|ou)\s*[0-9])+\b", re.IGNORECASE)
    res = nivel_trl_regex.search(text)
    if res:
        return res.group()
    else:
        single_num_regex = re.compile(r"\b[0-9]\b")
        res = single_num_regex.search(text)
        if res:
            return res.group()
        else:
            return 'Não encontrado'

def get_attachments_links(agency_name, edital_page_link):
    """
    Retrieves the links to attachment PDFs from a given edital page link for a specified agency.

    Args:
        agency_name (str): The name of the agency.
        edital_page_link (str): The URL of the edital page.

    Returns:
        list: A list of URLs to the attachment PDFs.

    Example:
        >>> agency_name = 'some_agency'
        >>> edital_page_link = 'https://example.com/edital'
        >>> attachments = get_attachments_links(agency_name, edital_page_link)
        >>> print(attachments)

    Note:
        This function reads the configuration settings for the specified agency from a configuration file.
        It then initializes a `PDFScraping` object with the agency's name, the edital page link, and other settings.
        The function searches for PDF links on the page and filters them to include only those that match the pattern for attachments (case-insensitive 'anexo').
    """
    config = ConfigParser()
    config.read(os.path.join(constants.CONFIG_PATH, constants.SITES_CONFIG_FILE))

    verify = config.getboolean(agency_name, 'verify')
    two_step_pdf_check = config.getboolean(agency_name, 'two_step_pdf_check')

    pdf_scraping = PDFScraping(
        name=agency_name, 
        host=edital_page_link, 
        selector=None, 
        depth=1, 
        verify=verify,
        two_step_pdf_check=two_step_pdf_check,
    )
    pdfs_links = pdf_scraping.get_pdfs_links()

    attachment_pattern_regex = re.compile(r'(anexo)', re.IGNORECASE)
    attachment_pdfs = [l.host for l in pdfs_links if attachment_pattern_regex.search(l.host)]

    return attachment_pdfs

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
    
    editals_links_saved = scraping_db.get_all()
    editals_info_extracted = editals_db.get_all()
    links_info_editals_extracted = [e["ds_link_pdf"] for e in editals_info_extracted]

    if len(editals_links_saved) > 0:
        for edital in tqdm(editals_links_saved):
            if edital['ds_link_pdf'] not in links_info_editals_extracted:
                try:
                    print(f"Extracting {edital['ds_agency'].upper()} -> {edital['ds_link_pdf']}")
                    infos = get_pdf_infos(
                        edital_path=edital['ds_link_pdf'], 
                        is_document_pdf=edital['is_document_pdf'], 
                        use_attachment_files=edital['use_attachment_files'],
                        list_edital_attachment=get_attachments_links(edital['ds_agency'], edital['ds_parent_link']) if edital['use_attachment_files'] else [],
                        model_to_use=model_to_use,
                    )

                    editals_db.insert_data(
                        ds_link_pdf=edital['ds_link_pdf'],
                        ds_agency=edital['ds_agency'].upper(),
                        ds_titulo=parse_title(infos['titulos']),
                        ds_titulo_completo = parse_full_title(infos['titulo_completo']),
                        ds_numero=parse_edital_number(infos['numero']),
                        ds_objetivo=parse_objetivo(infos['objetivo']),
                        ds_elegibilidade=parse_elegibilidade(infos['elegibilidade']),
                        ds_submission=parse_datetime(infos['submissao'], defaul_return="Fluxo contínuo"),
                        dt_submission=parse_datetime(infos['submissao'], defaul_return=None),
                        ds_financiamento=parse_money_value(infos['financiamento']),
                        ds_currency=parse_currency(infos['financiamento']),
                        nm_financing_value=parse_financing_value(infos['financiamento']),
                        ds_areas=parse_areas(infos['areas']),
                        ds_nivel_trl=parse_nivel_trl(infos['nivel_trl']),
                        is_document_pdf=edital['is_document_pdf']
                    )

                except Exception as e:
                    print(f"Error on pdf {edital['ds_link_pdf']} -> {e}")
                    pass
    else:
        raise Exception(f"No pdfs found in {scraping_db.table_name} table!")

    scraping_db.close()
    editals_db.close()
    
if __name__ == '__main__':
    extract_pdf_infos_db()