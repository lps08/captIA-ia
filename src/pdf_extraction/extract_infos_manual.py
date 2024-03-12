#%%
from src.pdf_extraction.pdf_text_extraction import extract_content
from src.ml_models.word2vec.similarity import get_most_similar_candidate
from transformers import pipeline
import re
import numpy as np
from deep_translator import GoogleTranslator
from src import constants

def get_targets():
    """
    Retrieves target keywords to perform similarity search.

    Returns:
        dict: Target keywords for each information category.

    Note:
        This function returns a dictionary where each key represents an information category, 
        and the corresponding value is a list of target keywords associated with that category.
        These keywords can be used for similarity search tasks from textual data.

    Example:
        >>> targets = get_targets()
    """
    return {
        'objetivo' : ['objeto', 'objetivo'],
        'elegibilidade' : ['elegibilidade', 'requisitos', 'critérios', 'critério', 'elegíveis', 'condições', 'seletivo', 'julgamento'],
    }

def get_queries():
    """
    Retrieves queries to perform question answers.

    Returns:
        dict: Queries for each information category.

    Note:
        This function returns a dictionary where each key represents an information category,
        and the corresponding value is a list of queries associated with that category.
        These queries can be used for question answers tasks.

    Example:
        >>> queries = get_queries()
    """
    return {
        "cronograma" : [
            "When is date limit for submission?",
            "When is the limit submission of proposal?",
            "When is the limit for submission?",
        ],
        "recurso" : [
            "How much is the value of the resource?",
            # "How much is the global values of resource financed?",
        ]
    }

def get_cronograma_section(document_content_dict:dict):
    """
    Extracts the section containing the schedule (cronograma) from a document content dictionary.

    Args:
        document_content_dict (dict): A dictionary containing document content blocks.

    Returns:
        str: The section containing the schedule, or an empty string if not found.

    Note:
        This function searches for the section containing the schedule (cronograma) from the provided document content.
        It looks for patterns matching dates and months in the content blocks of each section.
        The section with the most detected dates is considered the schedule section.
        If months are detected more frequently than dates, and their count is more than half of the total count,
        the section with the most detected months is considered the schedule section.
        If neither dates nor months are found, an empty string is returned.

    Example:
        >>> document_content = {'section1': ['block1', 'block2'], 'section2': ['block3', 'block4']}
        >>> cronograma_section = get_cronograma_section(document_content)
    """
    date_pattern = re.compile(r"\b(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})\b")
    month_pattern = re.compile(r"\b(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+\b")
    target_section_dates = (0, '')
    target_section_months = (0, '')

    for key in document_content_dict.keys():
        blocks = document_content_dict[key]

        dates = date_pattern.findall(' '.join(blocks))
        months = month_pattern.findall(' '.join(blocks))

        if len(dates) > target_section_dates[0]:
            target_section_dates = (len(dates), key)

        if len(months) > target_section_months[0]:
            target_section_months = (len(months), key)

    if target_section_dates[0] >= (target_section_months[0] // 2) and target_section_dates[1] != '':
        return document_content_dict[target_section_dates[1]]
    elif target_section_months[1] != '':
        return document_content_dict[target_section_months[1]]
    return ''

def get_funds_sections(document_content_dict:dict):
    """
    Extracts sections containing information about funds from a document content dictionary.

    Args:
        document_content_dict (dict): A dictionary containing document content blocks.

    Returns:
        list: Sections containing information about funds.

    Note:
        This function searches for sections containing information about funds from the provided document content.
        It uses a regular expression pattern to detect monetary values in each block of text within the document content.
        If any monetary value is found in a block, the entire block is considered a funds section.
        All blocks containing information about funds are collected and returned as a list.

    Example:
        >>> document_content = {'section1': ['block1', 'block2'], 'section2': ['block3', 'block4']}
        >>> funds_sections = get_funds_sections(document_content)
    """
    money_pattern = re.compile(r'([£\$€])\s*(\d+(?:[\.\,]\d+)*)|(\d+(?:[\.\,]\d+)*)\s*([£\$€]|(euros|R\$|reais))')
    texts = []

    for key in document_content_dict.keys():
        blocks = document_content_dict[key]

        if any(money_pattern.findall(b) for b in blocks):
            texts.extend(blocks)

    return texts

def get_submission_date(document_content_dict, nlp_model, queries):
    """
    Extracts the submission date from a document content dictionary using a natural language processing model.

    Args:
        document_content_dict (dict): A dictionary containing document content blocks.
        nlp_model (object): The natural language processing model used for question answering.
        queries (list): A list of queries used to extract the submission date.

    Returns:
        tuple: A tuple containing the extracted submission date and additional information.

    Note:
        This function extracts the submission date from the provided document content dictionary using a natural
        language processing model for question answering. It considers both types of dates (numeric and textual).
        If valid dates are found in the cronograma section of the document content, the function translates the
        extracted text to English and uses the provided queries to extract the submission date.
        The function returns a tuple containing the extracted submission date and additional information.

    Example:
        >>> document_content = {'section1': ['block1', 'block2'], 'section2': ['block3', 'block4']}
        >>> nlp_model = ...  # Initialize the NLP model
        >>> queries = ['When is the deadline for submission?', 'What is the submission deadline?']
        >>> additional_info, submission_date = get_submission_date(document_content, nlp_model, queries)
    """
    # consider both types of dates
    date_pattern = re.compile(r"\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4})|(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+\b")
    dates_text = get_cronograma_section(document_content_dict)
    dates_text = list(filter(lambda x: True if date_pattern.findall(x) else False, dates_text))
    dates_text_translated = [GoogleTranslator(source='auto', target='en').translate(t) for t in dates_text if len(t.split()) <= 15]
    dates_text_translated = '\n '.join(dates_text_translated)

    res = []

    if len(dates_text_translated) > 0:
        for q in queries:
            res.append(nlp_model({
                'question': q,
                'context' : dates_text_translated
            }))

        res_max_id = np.argmax([r['score'] for r in res])
        # print(res[res_max_id]['score'])
        return dates_text, f"Data de submissão: {res[res_max_id]['answer']}"
    
    return dates_text, ''

def get_max_fund_money_value(document_content_dict, nlp_model, queries):
    """
    Extracts the maximum fund money value from a document content dictionary using a natural language processing model.

    Args:
        document_content_dict (dict): A dictionary containing document content blocks.
        nlp_model (object): The natural language processing model used for question answering.
        queries (list): A list of queries used to extract the maximum fund money value.

    Returns:
        tuple: A tuple containing the extracted maximum fund money value and additional information.

    Note:
        This function extracts the maximum fund money value from the provided document content dictionary using a
        natural language processing model for question answering. It translates the extracted text to English and
        uses the provided queries to extract the maximum fund money value. The function returns a tuple containing
        the extracted maximum fund money value and additional information.

    Example:
        >>> document_content = {'section1': ['block1', 'block2'], 'section2': ['block3', 'block4']}
        >>> nlp_model = ...  # Initialize the NLP model
        >>> queries = ['What is the maximum fund amount?', 'What is the maximum fund value?']
        >>> additional_info, max_fund_value = get_max_fund_money_value(document_content, nlp_model, queries)
    """
    founds_text = get_funds_sections(document_content_dict)
    founds_text_translated = [GoogleTranslator(source='auto', target='en').translate(t) for t in founds_text]
    founds_text_translated = list(filter(lambda x: True if x else False, founds_text_translated))
    founds_text_translated = '\n '.join(founds_text_translated)

    res = []

    if len(founds_text_translated) > 0:
        for q in queries:
            res.append(nlp_model({
                'question': q,
                'context' : founds_text_translated,
            }))

        res_max_id = np.argmax([r['score'] for r in res])
        res = res[res_max_id]['answer']
        res = GoogleTranslator(source='auto', target='pt').translate(res)
        return founds_text, f"Recurso máximo: {res}"
    
    return founds_text, ''


def extract_infos(pdf_path, model, w2v_threshold=0.28):
    """
    Extracts information from a PDF document using a given model.

    Args:
        pdf_path (str): The path to the PDF file.
        model (object): The word embedding model used for similarity calculation.
        w2v_threshold (float): The threshold for word embedding similarity.

    Returns:
        dict: A dictionary containing extracted information from the PDF.
        
    Note:
        This function extracts information from a PDF document using a given model for similarity calculation.
        It extracts various sections such as title, objective, eligibility, submission date, and maximum fund value
        using predefined targets, queries, and natural language processing techniques.
        The function returns a dictionary containing the extracted information.

    Example:
        >>> pdf_path = 'document.pdf'
        >>> model = ...  # Initialize the word2vec model
        >>> extracted_info = extract_infos(pdf_path, model)
    """
    pdf_content_extracted = extract_content(pdf_path)
    sections = list(pdf_content_extracted.keys())
    targets = get_targets()
    queries = get_queries()
    nlp = pipeline('question-answering', model=constants.QA_MODEL_NAME, tokenizer=constants.QA_MODEL_NAME)
    content_extracted = {}
    
    title = sections.pop(0)
    objetivo_key = get_most_similar_candidate(model, targets['objetivo'], sections, threshold=w2v_threshold)
    elegibilidade_key = get_most_similar_candidate(model, targets['elegibilidade'], sections, threshold=w2v_threshold)
    date_text, submition_date = get_submission_date(pdf_content_extracted, nlp, queries['cronograma'])
    found_texts, max_found_value = get_max_fund_money_value(pdf_content_extracted, nlp, queries['recurso'])

    content_extracted['titulo'] = title
    content_extracted['objetivo'] = pdf_content_extracted[objetivo_key][0] if objetivo_key else 'Nao encontrado'
    content_extracted['elegibilidade'] = pdf_content_extracted[elegibilidade_key] if elegibilidade_key else 'Não encontrado'
    content_extracted['submissao'] = submition_date
    content_extracted['financiamento'] = max_found_value
    content_extracted['areas'] = ''
    
    return content_extracted