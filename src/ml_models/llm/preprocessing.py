#%%
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from langchain_core.documents import Document
import requests
import tempfile
import re

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
    """
    Loads and processes a PDF document, partitioning it into text elements using various strategies.

    Args:
        pdf_path (str): The URL or local path to the PDF file.
        extract_images (bool): Whether to extract images from the PDF. Defaults to False.
        infer_table_structure (bool): Whether to infer table structures in the PDF. Defaults to True.
        languages (list): List of languages for text extraction. Defaults to ['por'].
        chunking_strategy (str): The strategy for chunking text. Defaults to "by_title".
        strategy (str): The extraction strategy. Defaults to 'auto'.
        max_characters (int): Maximum number of characters per chunk. Defaults to 1000.
        new_after_n_chars (int): Create new chunk after this many characters. Defaults to 750.
        combine_text_under_n_chars (int): Combine text chunks under this many characters. Defaults to 500.

    Returns:
        list: A list of Document objects containing the extracted text content.
    
    Note:
        This function downloads a PDF file from the provided URL, processes it to extract text elements based on the
        specified chunking and extraction strategies, and returns a list of Document objects containing the text content.
        The `partition_pdf` function from the `unstructured` library is used for the extraction process, and the function
        supports various options for handling images, table structures, and different languages. The chunking strategy and
        character limits help in dividing the text into manageable parts for further processing.

    Example:
        >>> docs = load_doc_uns(pdf_path='http://example.com/sample.pdf')
        >>> for doc in docs:
        >>>     print(doc.page_content)
    """
    response = requests.get(pdf_path)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf_file:
        temp_pdf_file.write(response.content)
        temp_pdf_path = temp_pdf_file.name

        raw_pdf_elements = partition_pdf(
            filename=temp_pdf_path,
            extract_images_in_pdf=extract_images,
            infer_table_structure=infer_table_structure,
            languages=languages,
            chunking_strategy=chunking_strategy,
            strategy=strategy,
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars,
            combine_text_under_n_chars=combine_text_under_n_chars,
        )
    
    text_elements = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            if str(element) != '':
                text_elements.append(str(element.metadata.text_as_html))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            if str(element) != '':
                text_elements.append(str(element))

    docs = [Document(page_content=text) for text in text_elements]

    return docs

def load_html_page_content(
    html_page_link,
    chunking_strategy="by_title",
    max_characters=1000,
    new_after_n_chars=750,
    combine_text_under_n_chars=200,
    skip_headers_and_footers=True,
    infer_table_structure=True,
):
    """
    Loads and processes an HTML page, partitioning it into text elements using various strategies.

    Args:
        html_page_link (str): The URL of the HTML page.
        chunking_strategy (str): The strategy for chunking text. Defaults to "by_title".
        max_characters (int): Maximum number of characters per chunk. Defaults to 1000.
        new_after_n_chars (int): Create new chunk after this many characters. Defaults to 750.
        combine_text_under_n_chars (int): Combine text chunks under this many characters. Defaults to 200.
        skip_headers_and_footers (bool): Whether to skip headers and footers in the HTML content. Defaults to True.
        infer_table_structure (bool): Whether to infer table structures in the HTML content. Defaults to True.

    Returns:
        list: A list of Document objects containing the extracted text content.

    Note:
        This function downloads an HTML page from the provided URL, processes it to extract text elements based on the
        specified chunking and extraction strategies, and returns a list of Document objects containing the text content.
        The `partition_html` function from the `unstructured` library is used for the extraction process, and the function
        supports various options for handling headers, footers, table structures, and different chunking strategies. The
        chunking strategy and character limits help in dividing the text into manageable parts for further processing.

    Example:
        >>> docs = load_html_page_content(html_page_link='http://example.com/page.html')
        >>> for doc in docs:
        >>>     print(doc.page_content)
    """
    raw_pdf_elements = partition_html(
        url=html_page_link, 
        chunking_strategy=chunking_strategy, 
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars,
        skip_headers_and_footers=skip_headers_and_footers,
        infer_table_structure=infer_table_structure,
    )

    text_elements = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            if str(element) != '':
                text_elements.append(str(element.metadata.text_as_html))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            if str(element) != '':
                text_elements.append(str(element))

    docs = [Document(page_content=text) for text in text_elements]

    return docs

def load_doc_with_attachments(
    pdf_path,
    list_edital_attachment,
    use_unstructured,
    extract_images=False, 
    infer_table_structure=True, 
    languages=['por'], 
    chunking_strategy="by_title", 
    strategy='auto',
    max_characters=1000,
    new_after_n_chars=750,
    combine_text_under_n_chars=500,
):
    """
    Loads and processes a main PDF document and its attachments.

    Args:
        pdf_path (str): The path to the main PDF document.
        list_edital_attachment (list): A list of URLs or paths to attachment PDFs.
        use_unstructured (bool): Flag to determine whether to use unstructured loading.
        extract_images (bool, optional): Flag to extract images from the PDF. Defaults to False.
        infer_table_structure (bool, optional): Flag to infer table structure in the PDF. Defaults to True.
        languages (list, optional): List of languages for text extraction. Defaults to ['por'].
        chunking_strategy (str, optional): Strategy for chunking text. Defaults to "by_title".
        strategy (str, optional): Strategy for loading the document. Defaults to 'auto'.
        max_characters (int, optional): Maximum number of characters per chunk. Defaults to 1000.
        new_after_n_chars (int, optional): Number of characters after which a new chunk starts. Defaults to 750.
        combine_text_under_n_chars (int, optional): Combine text chunks if they are under this number of characters. Defaults to 500.

    Returns:
        list: A list of Document objects containing the text and metadata from the main PDF and its attachments.

    Note:
        This function can load documents using either an unstructured method or a structured method based on the `use_unstructured` flag.
        It first loads the main PDF document, then iterates through the list of attachment URLs or paths and loads each attachment.
        The function returns a list of Document objects containing the text and metadata from the main PDF and its attachments.

    Example:
        >>> pdf_path = 'path/to/main.pdf'
        >>> attachments = ['path/to/attachment1.pdf', 'path/to/attachment2.pdf']
        >>> docs = load_doc_with_attachments(pdf_path, attachments, use_unstructured=True)
        >>> for doc in docs:
        >>>     print(doc.page_content)
    """
    load_doc_method = load_doc_uns if use_unstructured else load_doc
    docs = load_doc_method(pdf_path)

    for link in list_edital_attachment:
        doc = load_doc_method(link)
        docs.extend(doc)
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
        use_attachment_files=False,
        list_edital_attachment=[],
        search_algorithm_type='mmr',
        k=70,
        fetch_k=100,
        create_spliter=True,
        chunk_size=1024,
        chunk_overlap=512,
    ):
    """
    Creates a retriever from a PDF document using specified parameters and embeddings.

    Args:
        pdf_path (str): The path to the PDF document.
        embeddings (object): The embeddings object used for creating the retriever.
        use_unstructured (bool): Whether to use the unstructured document loader. Defaults to True.
        use_attachment_files (bool): Whether to include attachments in the document loading process. Defaults to False.
        list_edital_attachment (list): List of attachment files to be included if use_attachment_files is True. Defaults to an empty list.
        search_algorithm_type (str): The type of search algorithm to use. Defaults to 'mmr'.
        k (int): The number of top-k results to retrieve. Defaults to 70.
        fetch_k (int): The number of documents to fetch in the initial search. Defaults to 100.
        create_spliter (bool): Whether to create a splitter for document chunking. Defaults to True.
        chunk_size (int): The size of the chunks in which the document is split. Defaults to 1024.
        chunk_overlap (int): The overlap between chunks. Defaults to 512.

    Returns:
        object: A retriever object configured with the specified parameters.

    Note:
        This function creates a retriever for a given PDF document using the specified parameters and embeddings. The retriever can be configured to use unstructured document loading, which supports advanced document processing and chunking strategies. The function also supports the inclusion of attachment files if specified. The `create_retriever` function is used to create the actual retriever object, which is then used to perform searches on the document content.

    Example:
        >>> retriever = create_retriever_from_pdf(
        >>>     pdf_path='path/to/document.pdf',
        >>>     embeddings=embeddings,
        >>>     use_unstructured=True,
        >>>     search_algorithm_type='mmr',
        >>>     k=70,
        >>>     fetch_k=100,
        >>>     chunk_size=1024,
        >>>     chunk_overlap=512,
        >>> )
        >>> results = retriever.retrieve(query="sample query")
    """
    if use_unstructured:
        doc = load_doc_with_attachments(pdf_path, list_edital_attachment, True) if use_attachment_files else load_doc_uns(pdf_path)
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
        doc = load_doc_with_attachments(pdf_path, False) if use_attachment_files else load_doc(pdf_path)
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

def create_retriever_from_html_page(
    html_page_link, 
    embeddings,
    search_algorithm_type='mmr',
    k=50,
    fetch_k=100,
    create_spliter=True,
):
    """
    Creates a retriever from an HTML page using specified parameters and embeddings.

    Args:
        html_page_link (str): The URL of the HTML page.
        embeddings (object): The embeddings object used for creating the retriever.
        search_algorithm_type (str): The type of search algorithm to use. Defaults to 'mmr'.
        k (int): The number of top-k results to retrieve. Defaults to 50.
        fetch_k (int): The number of documents to fetch in the initial search. Defaults to 100.
        create_spliter (bool): Whether to create a splitter for document chunking. Defaults to True.

    Returns:
        object: A retriever object configured with the specified parameters.

    Note:
        This function creates a retriever for a given HTML page using the specified parameters and embeddings. The retriever can be configured to use different search algorithms and can retrieve a specified number of top results. The `create_retriever` function is used to create the actual retriever object, which is then used to perform searches on the HTML page content.

    Example:
        >>> retriever = create_retriever_from_html_page(
        >>>     html_page_link='http://example.com/page.html',
        >>>     embeddings=embeddings,
        >>>     search_algorithm_type='mmr',
        >>>     k=50,
        >>>     fetch_k=100,
        >>>     create_spliter=True,
        >>> )
        >>> results = retriever.retrieve(query="sample query")
    """
    doc = load_html_page_content(html_page_link)
    retriever = create_retriever(
        doc, 
        embeddings, 
        search_algorithm_type, 
        k,
        fetch_k,
        False,
    )

    return retriever