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

def create_retriever(doc, embeddings, search_algorithm_type, k, fetch_k, create_spliter, regex_filter_list=[], chunk_size=None, chunk_overlap=None):
    """
    Creates a retriever object from the provided documents and embeddings.

    This function processes a list of documents by optionally splitting them into chunks, applying regular expression filters, and creating a retriever using the specified search algorithm and parameters.

    Args:
        doc (list): List of Document objects to be processed.
        embeddings (Embeddings): The embeddings model used to convert text chunks into vector representations.
        search_algorithm_type (str): The type of search algorithm to use (e.g., 'mmr' for Maximal Marginal Relevance).
        k (int): The number of top relevant documents to return during retrieval.
        fetch_k (int): The number of documents to initially fetch during the search process.
        create_spliter (bool): Whether to split documents into smaller chunks.
        regex_filter_list (list, optional): List of compiled regular expressions to filter the chunks. Defaults to an empty list.
        chunk_size (int, optional): The size of each text chunk, if splitting is enabled. Defaults to None.
        chunk_overlap (int, optional): The number of overlapping characters between chunks, if splitting is enabled. Defaults to None.

    Returns:
        retriever: A retriever object that can be used to find relevant documents based on the provided embeddings and search algorithm.

    Example:
        >>> docs = [Document(page_content="This is a sample document.")]
        >>> embeddings = SomeEmbeddingsModel()
        >>> retriever = create_retriever(docs, embeddings, search_algorithm_type='mmr', k=5, fetch_k=10, create_spliter=True, chunk_size=500, chunk_overlap=50)
    """
    if create_spliter:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(doc)
    else:
        chunks = doc
    
    filtered_chunks = []
    for regex in regex_filter_list:
        filtered_chunks.extend(list(filter(lambda x: True if regex.findall(x.page_content) else False, doc)))

    chunks = filtered_chunks if len(filtered_chunks) > 0 else chunks
        
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type=search_algorithm_type, search_kwargs={'k':k, 'fetch_k':fetch_k})
    return retriever

def create_retriever_from_pdf(
        pdf_path, 
        embeddings,
        use_unstructured=True,
        use_attachment_files=False,
        list_edital_attachment=[],
        regex_filter_list = [],
        search_algorithm_type='mmr',
        k=50,
        fetch_k=100,
        create_spliter=True,
        chunk_size=1024,
        chunk_overlap=512,
        n_first_docs = None,
    ):
    """
    Creates a retriever from a PDF document and optionally its attachments.

    This function processes a PDF document (and optionally its attachments) using either structured or unstructured methods, applies optional regular expression filters, and creates a retriever using the specified search algorithm and parameters.

    Args:
        pdf_path (str): The path to the PDF document.
        embeddings (Embeddings): The embeddings model used to convert text chunks into vector representations.
        use_unstructured (bool): Whether to use unstructured methods for document processing. Defaults to True.
        use_attachment_files (bool): Whether to process attachment files. Defaults to False.
        list_edital_attachment (list, optional): List of attachment file paths to include. Defaults to an empty list.
        regex_filter_list (list, optional): List of compiled regular expressions to filter the chunks. Defaults to an empty list.
        search_algorithm_type (str): The type of search algorithm to use (e.g., 'mmr' for Maximal Marginal Relevance). Defaults to 'mmr'.
        k (int): The number of top relevant documents to return during retrieval. Defaults to 50.
        fetch_k (int): The number of documents to initially fetch during the search process. Defaults to 100.
        create_spliter (bool): Whether to split documents into smaller chunks. Defaults to True.
        chunk_size (int, optional): The size of each text chunk, if splitting is enabled. Defaults to 1024.
        chunk_overlap (int, optional): The number of overlapping characters between chunks, if splitting is enabled. Defaults to 512.
        n_first_docs (int, optional): Number of first documents to consider. Defaults to None.

    Returns:
        retriever: A retriever object that can be used to find relevant documents based on the provided embeddings and search algorithm.

    Example:
        >>> embeddings = SomeEmbeddingsModel()
        >>> retriever = create_retriever_from_pdf("path/to/pdf", embeddings, search_algorithm_type='mmr', k=5, fetch_k=10)
    """
    if use_unstructured:
        doc = load_doc_with_attachments(pdf_path, list_edital_attachment, True) if use_attachment_files else load_doc_uns(pdf_path)
        doc = doc[:n_first_docs] if n_first_docs else doc
        retriever = create_retriever(
            doc, 
            embeddings, 
            search_algorithm_type, 
            k,
            fetch_k,
            False,
            regex_filter_list,
            chunk_size,
            chunk_overlap,
        )
    else:
        doc = load_doc_with_attachments(pdf_path, list_edital_attachment, False) if use_attachment_files else load_doc(pdf_path)
        doc = doc[:n_first_docs] if n_first_docs else doc
        retriever = create_retriever(
            doc, 
            embeddings, 
            search_algorithm_type, 
            k,
            fetch_k,
            create_spliter,
            regex_filter_list,
            chunk_size,
            chunk_overlap,
        )

    return retriever

def create_retriever_from_html_page(
    html_page_link, 
    embeddings,
    search_algorithm_type='mmr',
    regex_filter_list=[],
    k=50,
    fetch_k=100,
    create_spliter=True,
    n_first_docs = None,
):
    """
    Creates a retriever from an HTML page.

    This function processes an HTML page to extract content, applies optional regular expression filters, and creates a retriever using the specified search algorithm and parameters.

    Args:
        html_page_link (str): The URL of the HTML page.
        embeddings (Embeddings): The embeddings model used to convert text chunks into vector representations.
        search_algorithm_type (str): The type of search algorithm to use (e.g., 'mmr' for Maximal Marginal Relevance). Defaults to 'mmr'.
        regex_filter_list (list, optional): List of compiled regular expressions to filter the chunks. Defaults to an empty list.
        k (int): The number of top relevant documents to return during retrieval. Defaults to 50.
        fetch_k (int): The number of documents to initially fetch during the search process. Defaults to 100.
        create_spliter (bool): Whether to split documents into smaller chunks. Defaults to True.
        n_first_docs (int, optional): Number of first documents to consider. Defaults to None.

    Returns:
        retriever: A retriever object that can be used to find relevant documents based on the provided embeddings and search algorithm.

    Example:
        >>> embeddings = SomeEmbeddingsModel()
        >>> retriever = create_retriever_from_html_page("http://example.com/page", embeddings, search_algorithm_type='mmr', k=5, fetch_k=10)
    """
    doc = load_html_page_content(html_page_link)
    doc = doc[:n_first_docs] if n_first_docs else doc
    retriever = create_retriever(
        doc, 
        embeddings, 
        search_algorithm_type, 
        k,
        fetch_k,
        False,
        regex_filter_list,
    )

    return retriever

def get_general_info_retriever(
    edital,
    is_document_pdf,
    use_unstructured,
    use_attachment_files,
    list_edital_attachment,
    chunk_size,
    chunk_overlap,
    n_first_docs = 5,
    **kwargs,
):
    """
    Creates a retriever for general information extraction from an edital.

    This function processes an edital document (either PDF or HTML) and creates a retriever specifically designed for extracting general information, such as dates, locations, and deadlines. It utilizes regular expressions to filter relevant text chunks and applies appropriate settings for the retriever based on the document type.

    Args:
        edital (str): The path to the edital document (PDF or HTML).
        is_document_pdf (bool): Flag indicating whether the edital is in PDF format.
        n_first_docs (int, optional): Number of first documents to consider for the retriever. Defaults to 5.
        **kwargs: Additional keyword arguments to be passed to the underlying retriever creation functions.

    Returns:
        retriever: A retriever object optimized for extracting general information from the edital.

    Note:
        This function leverages the `create_retriever_from_pdf` and `create_retriever_from_html_page` functions to handle PDF and HTML edital formats, respectively. It applies specific regular expression filters to extract relevant text chunks and configures the retriever settings accordingly.

    Example:
        >>> edital_path = "path/to/edital.pdf"
        >>> retriever = get_general_info_retriever(edital_path, is_document_pdf=True)
        >>> # Use the retriever to find relevant information in the edital
    """
    if is_document_pdf:
        retriever = create_retriever_from_pdf(
            edital,
            n_first_docs=n_first_docs,
            use_unstructured=use_unstructured,
            use_attachment_files=use_attachment_files,
            list_edital_attachment=list_edital_attachment,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    else:
        retriever = create_retriever_from_html_page(
            edital,
            n_first_docs=n_first_docs,
            **kwargs,
        )

    return retriever

def get_values_info_retriever(
    edital,
    is_document_pdf,
    use_unstructured,
    use_attachment_files,
    list_edital_attachment,
    chunk_size,
    chunk_overlap,
    **kwargs
):
    """
    Creates a retriever for extracting values (dates and money) from an edital.

    This function processes an edital document (either PDF or HTML) and creates a retriever specifically designed for extracting values such as dates and monetary amounts. It utilizes regular expressions to filter relevant text chunks and applies appropriate settings for the retriever based on the document type.

    Args:
        edital (str): The path to the edital document (PDF or HTML).
        is_document_pdf (bool): Flag indicating whether the edital is in PDF format.
        **kwargs: Additional keyword arguments to be passed to the underlying retriever creation functions.

    Returns:
        retriever: A retriever object optimized for extracting values from the edital.

    Note:
        This function leverages the `create_retriever_from_pdf` and `create_retriever_from_html_page` functions to handle PDF and HTML edital formats, respectively. It applies specific regular expression filters to extract relevant text chunks and configures the retriever settings accordingly.

    Example:
        >>> edital_path = "path/to/edital.pdf"
        >>> retriever = get_values_info_retriever(edital_path, is_document_pdf=True)
    """
    money_pattern = re.compile(r'([£\$€])\s*(\d+(?:[\.\,]\d+)*)|(\d+(?:[\.\,]\d+)*)\s*([£\$€]|(euros|R\$|reais))')
    date_pattern = re.compile(r"\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4})|(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+\b")

    if is_document_pdf:
        retriever = create_retriever_from_pdf(
            edital,
            use_unstructured=use_unstructured,
            use_attachment_files=use_attachment_files,
            list_edital_attachment=list_edital_attachment,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            regex_filter_list=[date_pattern, money_pattern],
            **kwargs,
        )
    else:
        retriever = create_retriever_from_html_page(
            edital,
            regex_filter_list=[date_pattern, money_pattern],
            **kwargs,
        )

    return retriever