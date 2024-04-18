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
        use_unstructured=False,
        search_algorithm='mmr',
        k=50, 
        fetch_k=100, 
        create_spliter=True,
        chunk_size = 1024,
        chunk_overlap = 64,
    ):
    """
    Extracts information from a PDF document.

    This function extracts information from a PDF document using a combination of retriever and language model. 
    It formulates specific queries to retrieve relevant information such as the title, call number, objectives, 
    eligibility criteria, submission deadline, funding amount, areas of knowledge, and required technological 
    maturity level.

    Args:
        pdf_path (str): The path to the PDF file.
        llm (Model): The language model used for question answering.
        embeddings (Embeddings): The embeddings used for semantic search.
        parser (Parser): The parser used to extract information from the retrieved text.
        use_unstructured (bool): Whether to use unstructured search. Defaults to False.
        search_algorithm (str): The search algorithm used. Defaults to 'mmr'.
        k (int): The number of documents to retrieve in the initial search. Defaults to 50.
        fetch_k (int): The number of documents to fetch for detailed processing. Defaults to 100.
        create_spliter (bool): Whether to create a text spliter. Defaults to True.
        chunk_size (int): The size of text chunks for processing. Defaults to 1024.
        chunk_overlap (int): The overlap size between text chunks. Defaults to 64.

    Returns:
        dict: A dictionary containing the extracted information from the PDF.

    Raises:
        NoDateException: Raised if the submission deadline date is not found in the extracted information.

    Notes:
        This function utilizes retriever and language model components to extract information from a PDF document. 
        It formulates specific queries tailored to the desired information and retrieves relevant text passages 
        from the document. The extracted text is then parsed to extract structured information.

    Example:
        To extract information from a PDF document, you can use this function as follows:

        >>> pdf_path = 'document.pdf'
        >>> llm = LanguageModel()
        >>> embeddings = Embeddings()
        >>> parser = Parser()
        >>> info = extract_infos(pdf_path, llm, embeddings, parser)
        >>> print(info)

    """
    query = 'Extraia o título completo da chamada do documento. Extraia os titulo da chamada a partir do titulo completo. Qual o numero da chamada ou edital? Qual o objetivo da chamada? Liste os critérios de elegibilidade? Quando é a data deadline de submissão ou a data é de fluxo contínuo (sem data de submissão)? Quanto é o recurso financiado total (retorne Não encontrado se não conter o valor)? Liste as áreas de conhecimento? Qual o nível de maturidade tecnológica (TRL) necessário?'
    retriever = create_retriever_from_pdf(pdf_path, embeddings, use_unstructured, search_algorithm, k, fetch_k, create_spliter, chunk_size, chunk_overlap)
    res = qa_llm(query, llm, retriever, parser)
    res = parser.parse(res)
    if res.submissao == 'Não encontrado':
        raise NoDateException
    return res.dict()