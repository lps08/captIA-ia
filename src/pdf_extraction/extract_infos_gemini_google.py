from src.ml_models.llm.preprocessing import create_retriever_from_pdf
from src.ml_models.llm.retrieval_qa_llm import qa_llm
from google.generativeai.types.generation_types import StopCandidateException
from langchain_core.exceptions import OutputParserException
from retry import retry

class NoDateException(Exception):
    def __init__(self) -> None:
        super().__init__('No date found!')

@retry(
    tries=10, 
    delay=2, 
    backoff=1, 
    exceptions=(OutputParserException, StopCandidateException, NoDateException)
)
def extract_infos(
        pdf_path, 
        llm,
        embeddings,
        parser,
        search_algorithm='mmr',
        k=50, 
        fetch_k=100, 
        create_spliter=True,
        chunk_size = 1024,
        chunk_overlap = 128,
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
    query = "'Qual o título completo do documento? Qual o objetivo do edital? Quais todos os critérios de elegibilidade? Quando é a data deadline de submissão ou a data é de fluxo contínuo (sem data de submissão)? Quanto é o recurso financiado total (retorne Não encontrado se não conter o valor)? Quais as áreas da chamada? Qual o nível de maturidade tecnológica (TRL) necessário?'"
    retriever = create_retriever_from_pdf(pdf_path, embeddings, search_algorithm, k, fetch_k, create_spliter, chunk_size, chunk_overlap)
    res = qa_llm(query, llm, retriever, parser)
    res = parser.parse(res)
    if res.submissao == 'Não encontrado':
        raise NoDateException
    return res.dict()