#%%
from src.scraping.pdf_scraping import PDFScraping
from configparser import ConfigParser
import pandas as pd
from tqdm import tqdm
import requests
from io import BytesIO
from src.ml_models.bert.train import load_model
from src.ml_models.bert.predict import predict
from pdfminer.high_level import extract_text, extract_pages
from src import constants
import os

def get_pdf_links_from_agency(agency_name):
    """
    Retrieves PDF links from a specified agency's website based on configurations stored in a configuration file.

    Args:
        agency_name (str): The name of the agency for which PDF links are to be retrieved.

    Returns:
        list: A list of PDF links extracted from the agency's website.

    Note:
        This function reads configurations from a configuration file to determine the host, selector, depth, and verification status for scraping PDF links from the specified agency's website.
        It then utilizes a PDFScraping object to perform the scraping based on the provided parameters.
        Finally, it returns a list of PDF links extracted from the agency's website.

    Example:
        >>> agency_name = "example_agency"
        >>> get_pdf_links_from_agency(agency_name)
    """
    config = ConfigParser()
    config.read(os.path.join(constants.CONFIG_PATH, constants.SITES_CONFIG_FILE))

    host = config.get(agency_name, 'host')
    selector = config.get(agency_name, 'selector')
    depth = int(config.get(agency_name, 'depth'))
    verify = bool(config.get(agency_name, 'verify'))

    pdf_scraping = PDFScraping(agency_name, host, selector, depth, verify)
    pdfs_links = pdf_scraping.get_pdfs_links()

    return pdfs_links

def save_pdf_links_to_csv(pdfs_list, out_csv_file='pdfs-editais.csv'):
    """
    Saves a list of PDF objects to a CSV file containing their attributes (name, host, date).

    Args:
        pdfs_list (list): A list of PDF objects.
        out_csv_file (str, optional): The name of the output CSV file. Defaults to 'pdfs-editais.csv'.

    Returns:
        None

    Example:
        >>> pdf1 = PDF(name='pdf1.pdf', host='example.com', date='2022-01-01')
        >>> pdf2 = PDF(name='pdf2.pdf', host='example.com', date='2022-01-02')
        >>> pdfs_list = [pdf1, pdf2]
        >>> save_pdf_links_to_csv(pdfs_list, 'pdfs.csv')
    """
    pdfs_dataframe = pd.DataFrame(
        {
            'name': [pdf.name for pdf in pdfs_list],
            'host': [pdf.host for pdf in pdfs_list],
            'date': [pdf.date for pdf in pdfs_list],
        }
    )
    pdfs_dataframe.to_csv(out_csv_file)

def get_pdf_total_pages(pdf_path):
    """
    Calculates the total number of pages in a PDF document.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        int: The total number of pages in the PDF document.

    Example:
        >>> pdf_path = 'example.pdf'
        >>> total_pages = get_pdf_total_pages(pdf_path)
        >>> print(total_pages)
    """
    pages = list(extract_pages(pdf_path))
    return len(pages)

def get_editais_from_agency(agency_name, num_labels=2, max_content_lenght = 3000000, edital_threshold = 0.90, min_num_pages = 4):
    """
    Extracts "editais" (calls for proposals) from an agency's website.

    Args:
        agency_name (str): The name of the agency.
        num_labels (int): The number of labels for the BERT model.
        max_content_length (int): The maximum content length of a PDF file to consider.
        edital_threshold (float): The threshold probability for classifying a document as an "edital".
        min_num_pages (int): The minimum number of pages for a PDF document to be considered.

    Returns:
        list: A list of PDF objects representing the "editais" extracted from the agency's website.

    Example:
        >>> agency_name = 'example_agency'
        >>> editais = get_editais_from_agency(agency_name)
        >>> for edital in editais:
        >>>     print(edital.host)
    """
    pdfs_links = get_pdf_links_from_agency(agency_name)
    model, tokenizer = load_model(constants.BERT_MODEL_NAME, num_labels)
    editals = []

    for pdf in tqdm(pdfs_links):
        try:
            response = requests.get(pdf.host)

            if response.status_code == 200 and int(response.headers['Content-Length']) < max_content_lenght:
                pdf_bytes = BytesIO(response.content)
                doc_num_pages = get_pdf_total_pages(pdf_bytes)
                document = extract_text(pdf_bytes)

                if document and doc_num_pages >= min_num_pages:
                    predicted_class, probabilities = predict(
                        model=model,
                        tokenizer=tokenizer,
                        document_text=document,
                    )                    
                    if predicted_class == 1 and probabilities[0][predicted_class] > edital_threshold:
                        editals.append(pdf)
                else:
                    raise Exception(f"PDF {pdf.host} contais few pages {doc_num_pages}!")
        except Exception as e:
            # print(f"Error: {e}")
            pass

    return editals

if __name__ == '__main__':
    agency = "funcap"
    editais = get_editais_from_agency(
        agency,
    )

    for i in editais:
        print(i.host)