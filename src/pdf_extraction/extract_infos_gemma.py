from src.ml_models.llm.preprocessing import load_doc, create_retriever
from src.ml_models.llm.retrieval_qa_llm import qa_llm

def extract_infos(
        pdf_path, 
        llm, 
        embeddings, 
        n_first_p=3, 
        search_algorithm='mmr',
        k=6, 
        fetch_k=50, 
        create_spliter=False
    ):
    """
    Extracts various information from the specified sections of a PDF document.

    Args:
        pdf_path (str): The path to the PDF document.
        llm: The language model used for question answering.
        embeddings: The embeddings used for retrieval.
        n_first_p (int, optional): The number of first paragraphs to consider. Defaults to 3.
        search_algorithm (str, optional): The type of search algorithm. Defaults to 'mmr'.
        k (int, optional): The number of top documents to retrieve. Defaults to 6.
        fetch_k (int, optional): The number of documents to fetch from the retriever. Defaults to 50.
        create_spliter (bool, optional): Whether to create a text splitter. Defaults to False.

    Returns:
        dict: Extracted information including title, objective, submission deadline, eligibility criteria,
              financial resources, and areas of knowledge.

    Note:
        This function extracts various information from the specified sections of a PDF document.
        It loads the document, extracts the first paragraphs, and creates retrievers for these paragraphs and the entire document.
        Then, it uses the retrievers to retrieve relevant information for each section of interest.
        The extracted information is stored in a dictionary and returned.

    Example:
        >>> pdf_path = "path/to/pdf_document.pdf"
        >>> llm = get_language_model()
        >>> embeddings = get_embeddings()
        >>> extracted_info = extract_infos(pdf_path, llm, embeddings)
    """
    doc = load_doc(pdf_path)
    first_p_doc = doc[:n_first_p]
    first_p_retriever = create_retriever(first_p_doc, embeddings, search_algorithm, k, fetch_k, create_spliter)
    all_p_retriever = create_retriever(doc, embeddings, search_algorithm, k, fetch_k, create_spliter)
    edital = {}

    edital['titulo'] = qa_llm('Qual o titulo da chamada', llm, first_p_retriever)
    edital['objetivo'] = qa_llm('Qual o objetivo da chamada?', llm, first_p_retriever)
    edital['submissao'] = qa_llm('Qual é a data do período limite de inscrição?', llm, all_p_retriever)
    edital['elegibilidade'] = qa_llm('Liste os critérios de elegibilidade da chamada', llm, all_p_retriever)
    edital['recurso'] = qa_llm('Qual o valor dos recursos financeiro da chamada?', llm, all_p_retriever)
    edital['areas'] = qa_llm('Quais as todas áreas de conhecimento da chamada?', llm, all_p_retriever)

    return edital