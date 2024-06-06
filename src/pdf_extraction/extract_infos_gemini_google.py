from src.ml_models.llm.preprocessing import create_retriever_from_pdf, create_retriever_from_html_page
from src.ml_models.llm.retrieval_qa_llm import qa_llm
from src.ml_models.llm.base_models import edital_model
from google.generativeai.types.generation_types import StopCandidateException
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import PydanticOutputParser
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
    Extracts detailed information from a document using a specified language model and embeddings.

    This function processes a document (either PDF or HTML) to extract specific information based on a query. It utilizes a retriever to gather relevant document chunks and then uses a language model to generate the required information.

    Args:
        pdf_path (str): The path or URL to the document.
        llm: The language model to use for processing.
        embeddings: The embeddings to use for creating the retriever.
        is_document_pdf (bool): Indicates whether the document is a PDF.
        use_attachment_files (bool): Whether to use additional attachment files associated with the document.
        list_edital_attachment (list, optional): A list of attachment file paths or URLs. Defaults to an empty list.
        use_unstructured (bool, optional): Whether to use unstructured data processing. Defaults to False.
        search_algorithm (str, optional): The search algorithm to use ('mmr' by default). Defaults to 'mmr'.
        k (int, optional): The number of top documents to retrieve. Defaults to 100.
        fetch_k (int, optional): The number of documents to fetch. Defaults to 150.
        create_spliter (bool, optional): Whether to create a text splitter. Defaults to True.
        chunk_size (int, optional): The size of text chunks to split the document into. Defaults to 1024.
        chunk_overlap (int, optional): The overlap size between text chunks. Defaults to 64.

    Returns:
        dict: A dictionary containing the extracted information.

    Example:
        >>> pdf_path = "/path/to/document.pdf"
        >>> llm = get_some_llm_model()
        >>> embeddings = get_some_embeddings()
        >>> infos = extract_infos(
                pdf_path, llm, embeddings, 
                is_document_pdf=True, 
                use_attachment_files=True, 
                list_edital_attachment=["/path/to/attachment1.pdf"]
            )
        >>> print(infos)
    """
    query = 'Extraia o título da chamada/edital completo do documento. Extraia os titulo da chamada/edital a partir do titulo completo do documento. Qual o numero da chamada ou edital, se possuir? Qual o objetivo da chamada? Liste os critérios de elegibilidade? Qual a data inicial de lançamento da chamada/edital? Quando é a data deadline de submissão ou a data é de fluxo contínuo (sem data de submissão)? Quanto é o recurso financiado total/maximo com o tipo de moeda (retorne Não encontrado se não conter o valor)? Liste as áreas de conhecimento da chamada/edital? Qual o nível de maturidade tecnológica (TRL) necessário?'
    if is_document_pdf:
        retriever = create_retriever_from_pdf(edital_path, embeddings, use_unstructured, use_attachment_files, list_edital_attachment, [], search_algorithm, k, fetch_k, create_spliter, chunk_size, chunk_overlap)
    else:
        retriever = create_retriever_from_html_page(edital_path, embeddings, search_algorithm, [],  k, fetch_k)

    edital_parser = PydanticOutputParser(pydantic_object=edital_model.Edital)
    res = qa_llm(query, llm, retriever, edital_parser)
    res = edital_parser.parse(res)
    return res.dict()