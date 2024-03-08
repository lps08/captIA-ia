#%%
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.exceptions import OutputParserException
from google.generativeai.types.generation_types import StopCandidateException
from retry import retry
from src.ml_models.llm.retrieval_qa_llm import qa_llm
from src.ml_models.llm.preprocessing import create_retriever_from_pdf
from langchain.output_parsers import PydanticOutputParser
from src.ml_models.llm.base_models.edital_model import Edital
from src import constants

def disable_safety_settings():
    """
    Disable safety settings for various harm categories.

    Returns:
        dict: A dictionary mapping harm categories to harm block thresholds set to BLOCK_NONE.

    Note:
        This function returns a dictionary specifying harm categories with their corresponding harm block thresholds set to BLOCK_NONE.
        Disabling safety settings for these categories allows content related to these categories to pass through without blocking.

    Example:
        >>> settings = disable_safety_settings()
        >>> print(settings)
        {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
        }
    """
    return {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }

def get_parser():
    """
    Get an instance of PydanticOutputParser initialized with a Pydantic object.

    Returns:
        PydanticOutputParser: An instance of PydanticOutputParser initialized with a Pydantic object.

    Note:
        This function returns an instance of PydanticOutputParser initialized with a specific Pydantic object (e.g., Edital).
        The returned PydanticOutputParser instance can be used to parse and validate data according to the Pydantic object's schema.

    Example:
        >>> parser = get_parse()
        >>> parsed_data = parser.parse(raw_data)
    """
    return PydanticOutputParser(pydantic_object=Edital)

def get_gemini_model(
        name=constants.GOOGLE_GEMINI_MODEL_NAME, 
        temperature=0.2, 
        convert_system_message_to_human=True, 
        disable_safety=True
    ):
    """
    Get an instance of the Google Gemini generative AI model with specified settings.

    Args:
        name (str, optional): The name of the Google Gemini model. Defaults to constants.GOOGLE_GEMINI_MODEL_NAME.
        temperature (float, optional): The temperature parameter controlling the randomness of responses. Defaults to 0.2.
        convert_system_message_to_human (bool, optional): Whether to convert system messages to human-readable format. Defaults to True.
        disable_safety (bool, optional): Whether to disable safety settings for harmful content. Defaults to True.

    Returns:
        ChatGoogleGenerativeAI: An instance of the Google Gemini generative AI model.

    Note:
        This function returns an instance of the Google Gemini generative AI model with specified settings.
        It allows customization of model name, temperature, conversion of system messages, and disabling safety settings.

    Example:
        >>> gemini_model = get_gemini_model()
        >>> response = gemini_model.invoke("Hello!")
        >>> print(response)
        "Hi there!"
    """
    llm = ChatGoogleGenerativeAI(
        model=name, 
        temperature=temperature, 
        convert_system_message_to_human=convert_system_message_to_human, 
        safety_settings=disable_safety_settings() if disable_safety else None
    )
    return llm

@retry(
    tries=8, 
    delay=2, 
    backoff=1, 
    exceptions=(OutputParserException, StopCandidateException)
)
def extract_info_edital(
        pdf_path, 
        llm,
        embeddings,
        parser = PydanticOutputParser(pydantic_object=Edital),
        search_algorithm='mmr',
        k=50, 
        fetch_k=100, 
        create_spliter=True,
        chunk_size = 1024,
        chunk_overlap = 512,
    ):
    """
    Extract information from a PDF document (Edital) using a generative language model and embeddings.

    Args:
        pdf_path (str): The path to the PDF document.
        llm: The generative language model for question answering.
        embeddings: The embeddings used for text retrieval.
        parser: The output parser for parsing the extracted information. Defaults to PydanticOutputParser initialized with Edital schema.
        search_algorithm (str, optional): The text retrieval algorithm. Defaults to 'mmr'.
        k (int, optional): The number of top-ranked candidates returned by the retriever. Defaults to 50.
        fetch_k (int, optional): The number of documents fetched by the retriever. Defaults to 100.
        create_spliter (bool, optional): Whether to create a spliter during text retrieval. Defaults to True.
        chunk_size (int, optional): The size of text chunks used during retrieval. Defaults to 1024.
        chunk_overlap (int, optional): The overlap size between text chunks during retrieval. Defaults to 512.

    Returns:
        Any: Extracted information from the PDF document.

    Raises:
        OutputParserException: If an exception occurs during output parsing.
        StopCandidateException: If there are no more candidates to process during retrieval.

    Note:
        This function extracts information from a PDF document (Edital) using a generative language model and embeddings.
        It utilizes a retry decorator to retry the function in case of exceptions specified in the exceptions parameter.
        The pdf_path parameter specifies the path to the PDF document to be processed.
        The llm parameter represents the generative language model used for question answering.
        The embeddings parameter represents the embeddings used for text retrieval.
        The parser parameter specifies the output parser for parsing the extracted information.
        The search_algorithm parameter specifies the text retrieval algorithm to be used.
        Additional parameters control the behavior of the text retriever.

    Example:
        >>> pdf_path = 'path/to/pdf_document.pdf'
        >>> llm = get_gemini_model()
        >>> embeddings = get_embeddings_model()
        >>> extracted_info = extract_info_edital(pdf_path, llm, embeddings)
    """
    query = 'Qual o título completo do documento? Qual o objetivo do edital? Quais todos os critérios de elegibilidade? Quando é a data deadline de submissão? Quanto é o recurso financiado total? Quais todas as áreas de conhecimento da chamada?',
    retriever = create_retriever_from_pdf(pdf_path, embeddings, search_algorithm, k, fetch_k, create_spliter, chunk_size, chunk_overlap)
    res = qa_llm(query, llm, retriever, parser)

    return res