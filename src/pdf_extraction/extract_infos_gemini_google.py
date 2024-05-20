from src.ml_models.llm.preprocessing import create_retriever_from_pdf, create_retriever_from_html_page
from src.ml_models.llm.retrieval_qa_llm import qa_llm
from google.generativeai.types.generation_types import StopCandidateException
from langchain_core.exceptions import OutputParserException
from retry import retry

class NoDateException(Exception):
    def __init__(self) -> None:
        super().__init__('No date found!')

@retry(
    tries=5, 
    delay=2, 
    backoff=2, 
    exceptions=(OutputParserException, StopCandidateException)
)
def extract_infos(
        edital_path, 
        llm,
        embeddings,
        parser,
        is_document_pdf,
        use_attachment_files,
        list_edital_attachment=[],
        use_unstructured=False,
        search_algorithm='mmr',
        k=100, 
        fetch_k=150, 
        create_spliter=True,
        chunk_size = 1024,
        chunk_overlap = 64,
    ):
    """
    Extracts information from a document (PDF or HTML) related to a call for proposals or an edital.

    Args:
        edital_path (str): The path or URL to the document.
        llm: The language model to use for processing queries.
        embeddings: The embeddings model to use for document retrieval.
        parser: The parser to use for interpreting the model's output.
        is_document_pdf (bool): Flag indicating if the document is a PDF.
        use_attachment_files (bool, optional): Flag to determine if attachments should be used. Defaults to False.
        list_edital_attachment (list, optional): List of URLs or paths to attachment PDFs. Defaults to [].
        use_unstructured (bool, optional): Flag to determine whether to use unstructured loading for PDFs. Defaults to False.
        search_algorithm (str, optional): The search algorithm to use for document retrieval. Defaults to 'mmr'.
        k (int, optional): The number of top documents to consider during retrieval. Defaults to 100.
        fetch_k (int, optional): The number of documents to fetch initially. Defaults to 150.
        create_spliter (bool, optional): Flag to determine if a text splitter should be created. Defaults to True.
        chunk_size (int, optional): The size of text chunks to use. Defaults to 1024.
        chunk_overlap (int, optional): The overlap between text chunks. Defaults to 64.

    Returns:
        dict: A dictionary containing extracted information from the document.

    Note:
        This function is decorated with a retry mechanism that attempts to run the function up to 5 times with an exponential backoff.
        It uses a query to extract specific information such as the title, number, objectives, eligibility criteria, submission deadline,
        funding amount, knowledge areas, and technology readiness level (TRL) from the document.

    Example:
        >>> edital_path = 'path/to/document.pdf'
        >>> llm = get_llm_model()
        >>> embeddings = get_embeddings_model()
        >>> parser = get_parser()
        >>> result = extract_infos(
        >>>     edital_path,
        >>>     llm,
        >>>     embeddings,
        >>>     parser,
        >>>     is_document_pdf=True,
        >>>     use_attachment_files=True,
        >>>     list_edital_attachment=['path/to/attachment1.pdf', 'path/to/attachment2.pdf']
        >>> )
        >>> print(result)
    """
    query = 'Extraia o título da chamada/edital completo do documento. Extraia os titulo da chamada/edital a partir do titulo completo do documento. Qual o numero da chamada ou edital, se possuir? Qual o objetivo da chamada? Liste os critérios de elegibilidade? Qual a data inicial de lançamento da chamada/edital? Quando é a data deadline de submissão ou a data é de fluxo contínuo (sem data de submissão)? Quanto é o recurso financiado total/maximo (retorne Não encontrado se não conter o valor)? Liste as áreas de conhecimento da chamada/edital? Qual o nível de maturidade tecnológica (TRL) necessário?'
    if is_document_pdf:
        retriever = create_retriever_from_pdf(edital_path, embeddings, use_unstructured, use_attachment_files, list_edital_attachment, search_algorithm, k, fetch_k, create_spliter, chunk_size, chunk_overlap)
    else:
        retriever = create_retriever_from_html_page(edital_path, embeddings, search_algorithm, k, fetch_k)

    res = qa_llm(query, llm, retriever, parser)
    res = parser.parse(res)
    return res.dict()