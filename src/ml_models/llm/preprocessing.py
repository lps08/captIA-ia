#%%
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

def load_doc(pdf_path):
    """
    Load a PDF document and return its content.

    Args:
        pdf_path (str): The path to the PDF document.

    Returns:
        list: A list of document object representing the content of the PDF document.

    Note:
        This function loads a PDF document using PyPDFLoader and returns its content as a list of documents objects.
        Each string in the list represents a page in the PDF document.

    Example:
        >>> pdf_path = 'path/to/pdf_document.pdf'
        >>> document_content = load_doc(pdf_path)
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs

def load_doc_uns(
        pdf_path, 
        extract_images=False, 
        infer_table_structure=True, 
        languages=['por'], 
        chunking_strategy="by_title", 
        strategy='auto',
        max_characters=1000,
        new_after_n_chars=750,
        combine_text_under_n_chars=500,
    ):
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=extract_images,
        infer_table_structure=infer_table_structure,
        languages=languages,
        chunking_strategy=chunking_strategy,
        strategy=strategy,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars,
    )
    
    table_elements = []
    text_elements = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            if str(element) != '':
                table_elements.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            if str(element) != '':
                text_elements.append(str(element))

    docs = [Document(page_content=text) for text in table_elements + text_elements]

    return docs

def create_retriever(doc, embeddings, search_algorithm_type, k, fetch_k, create_spliter, chunk_size=None, chunk_overlap=None):
    """
    Create a retriever for text retrieval.

    Args:
        doc: The document or chunks of the document to be indexed for retrieval.
        embeddings: The embeddings used for vectorization.
        search_algorithm_type (str): The type of search algorithm to be used for retrieval.
        k (int): The number of top-ranked candidates to retrieve.
        fetch_k (int): The number of documents to fetch.
        create_spliter (bool): Whether to create a text splitter for chunking the document.
        chunk_size (int, optional): The size of text chunks. Required if create_spliter is True.
        chunk_overlap (int, optional): The overlap size between text chunks. Required if create_spliter is True.

    Returns:
        retriever: A retriever object for text retrieval.

    Note:
        This function creates a retriever object for text retrieval.
        It allows the indexing of a document or chunks of a document for retrieval.
        The retrieval process utilizes a specified search algorithm, such as MMR (Maximal Marginal Relevance).
        Additional parameters control the behavior of the retriever, including the number of candidates to retrieve and fetch.

    Example:
        >>> document = load_doc(pdf_path)
        >>> embeddings = get_embeddings_model()
        >>> retriever = create_retriever(document, embeddings, 'mmr', 10, 50, False)
    """
    if create_spliter:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(doc)
    else:
        chunks = doc
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type=search_algorithm_type, search_kwargs={'k':k, 'fetch_k':fetch_k})
    return retriever

def create_retriever_from_pdf(
        pdf_path, 
        embeddings,
        use_unstructured=True,
        search_algorithm_type='mmr',
        k=50,
        fetch_k=100,
        create_spliter=True,
        chunk_size=1024,
        chunk_overlap=512,
    ):
    """
    Create a retriever for text retrieval from a PDF document.

    Args:
        pdf_path (str): The path to the PDF document.
        embeddings: The embeddings used for vectorization.
        search_algorithm_type (str, optional): The type of search algorithm to be used for retrieval. Defaults to 'mmr'.
        k (int, optional): The number of top-ranked candidates to retrieve. Defaults to 50.
        fetch_k (int, optional): The number of documents to fetch. Defaults to 100.
        create_spliter (bool, optional): Whether to create a text splitter for chunking the document. Defaults to True.
        chunk_size (int, optional): The size of text chunks. Required if create_spliter is True. Defaults to 1024.
        chunk_overlap (int, optional): The overlap size between text chunks. Required if create_spliter is True. Defaults to 512.

    Returns:
        retriever: A retriever object for text retrieval.

    Note:
        This function creates a retriever object for text retrieval from a PDF document.
        It loads the PDF document, retrieves its content, and creates a retriever based on the specified parameters.
        The retrieval process utilizes a specified search algorithm, such as MMR (Maximal Marginal Relevance).
        Additional parameters control the behavior of the retriever, including the number of candidates to retrieve and fetch.

    Example:
        >>> pdf_path = 'path/to/pdf_document.pdf'
        >>> embeddings = get_embeddings_model()
        >>> retriever = create_retriever_from_pdf(pdf_path, embeddings)
    """
    if use_unstructured:
        doc = load_doc_uns(pdf_path)
        retriever = create_retriever(
            doc, 
            embeddings, 
            search_algorithm_type, 
            k,
            fetch_k,
            False,
            chunk_size,
            chunk_overlap,
        )
    else:
        doc = load_doc(pdf_path)
        retriever = create_retriever(
            doc, 
            embeddings, 
            search_algorithm_type, 
            k,
            fetch_k,
            create_spliter,
            chunk_size,
            chunk_overlap,
        )

    return retriever