from src.ml_models.llm.preprocessing import create_retriever_from_pdf
from src.ml_models.llm.retrieval_qa_llm import qa_llm

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
        chunk_overlap = 512,
    ):
    """
    Extracts various information from a PDF document using a language model, embeddings, and a parser.

    Args:
        pdf_path (str): The path to the PDF document.
        llm: The language model used for question answering.
        embeddings: The embeddings used for retrieval.
        parser: The parser object used for parsing the answer.
        search_algorithm (str, optional): The type of search algorithm. Defaults to 'mmr'.
        k (int, optional): The number of top documents to retrieve. Defaults to 50.
        fetch_k (int, optional): The number of documents to fetch from the retriever. Defaults to 100.
        create_spliter (bool, optional): Whether to create a text spliter. Defaults to True.
        chunk_size (int, optional): The size of each chunk for splitting text. Defaults to 1024.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 512.

    Returns:
        dict: The extracted information.

    Note:
        This function extracts various information from a PDF document using a language model, embeddings, and a parser.
        It constructs a query containing multiple questions about the document.
        Then, it creates a retriever using the provided embeddings and search parameters.
        The retriever is used to retrieve relevant documents from the PDF.
        The language model is applied to answer the query based on the retrieved documents.
        Finally, the parser is used to parse the answer and extract structured information.

    Example:
        >>> pdf_path = "path/to/pdf_document.pdf"
        >>> llm = get_language_model()
        >>> embeddings = get_embeddings()
        >>> parser = get_parser()
        >>> extracted_info = extract_infos(pdf_path, llm, embeddings, parser)
    """
    query = 'Qual o título completo do documento? Qual o objetivo do edital? Quais todos os critérios de elegibilidade? Quando é a data deadline de submissão? Quanto é o recurso financiado total? Quais todas as áreas de conhecimento da chamada?',
    retriever = create_retriever_from_pdf(pdf_path, embeddings, search_algorithm, k, fetch_k, create_spliter, chunk_size, chunk_overlap)
    res = qa_llm(query, llm, retriever, parser)

    return res