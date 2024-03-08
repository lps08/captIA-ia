#%%
import sys
sys.path.append("../../../")
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from src.pdf_extraction.pdf_text_extraction import extract_content
import os
import re
from unidecode import unidecode

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content.

    Note:
        This function extracts text content from a PDF file using the extract_content function.
        It then joins the keys of the extracted content dictionary into a single string and returns it.

    Example:
        >>> pdf_path = "path/to/pdf_file.pdf"
        >>> text_content = extract_text_from_pdf(pdf_path)
    """
    content = extract_content(pdf_path)
    return ' '.join(content.keys())

def tokenize_text(text):
    """
    Tokenizes the given text using NLTK's word_tokenize function.

    Args:
        text (str): The text to be tokenized.

    Returns:
        list: A list of tokens.

    Note:
        This function tokenizes the given text using NLTK's word_tokenize function.

    Example:
        >>> text = "This is a sample text for tokenization."
        >>> tokens = tokenize_text(text)
    """
    tokens = word_tokenize(text)
    return tokens

def preprocess_text(text, lang='portuguese'):
    """
    Preprocesses the given text by converting to lowercase, removing non-alphabetic characters,
    removing stopwords, and normalizing accents.

    Args:
        text (str): The text to be preprocessed.
        lang (str, optional): The language of the text. Defaults to 'portuguese'.

    Returns:
        str: The preprocessed text.
        
    Note:
        This function preprocesses the given text by performing the following steps:
        1. Converts the text to lowercase.
        2. Removes non-alphabetic characters and replaces them with spaces, except for periods.
        3. Normalizes accents to their ASCII equivalents.
        4. Tokenizes the text and removes stopwords specific to the specified language.
        5. Joins the remaining tokens into a single string.

    Example:
        >>> text = "Este é um exemplo de texto para pré-processamento."
        >>> preprocessed_text = preprocess_text(text)
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.]', '', text)

    text = unidecode(text)

    tokens = word_tokenize(text, language=lang)
    stop_words = set(stopwords.words(lang))
    tokens = [word for word in tokens if word not in stop_words and word != '9']
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def load_docs_dataset(pdfs_directory):
    """
    Load a dataset of documents from PDF files in the specified directory.

    Args:
        pdfs_directory (str): The path to the directory containing PDF files.

    Returns:
        list: A list of tokenized documents.

    Note:
        This function loads a dataset of documents from PDF files in the specified directory.
        It iterates through all PDF files in the directory, extracts text from each PDF,
        preprocesses the text, tokenizes it, and appends the tokens to the corpus.

    Example:
        >>> pdfs_directory = "/path/to/pdf_directory"
        >>> dataset = load_docs_dataset(pdfs_directory)
    """
    corpus = []
    
    for pdf_file in os.listdir(pdfs_directory):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdfs_directory, pdf_file)
            text = extract_text_from_pdf(pdf_path)

            preprocessed_text = preprocess_text(text)
            tokens = tokenize_text(preprocessed_text)
            corpus.append(tokens)

    return corpus

def train_fine_tune(model_path, dataset, out_fine_tuned_model):
    """
    Fine-tunes a pre-trained Word2Vec model with additional data from a given dataset.

    Args:
        model_path (str): The path to the pre-trained Word2Vec model.
        dataset (list): The dataset used for fine-tuning.
        out_fine_tuned_model (str): The path to save the fine-tuned Word2Vec model.

    Returns:
        None

    Note:
        This function fine-tunes a pre-trained Word2Vec model with additional data from the given dataset.
        It loads the pre-trained model, updates its vocabulary with the dataset, and continues training on the dataset.
        Finally, it saves the fine-tuned model to the specified output path.

    Example:
        >>> model_path = "path/to/pretrained_model.bin"
        >>> dataset = [["example", "text", "for", "fine-tuning"], ["another", "example"]]
        >>> out_fine_tuned_model = "path/to/fine_tuned_model.bin"
        >>> train_fine_tune(model_path, dataset, out_fine_tuned_model)
    """
    pretrained_model = KeyedVectors.load_word2vec_format(model_path)
    pretrained_model.build_vocab(dataset, update=True)
    pretrained_model.train(dataset, total_examples=pretrained_model.corpus_count, epochs=pretrained_model.epochs)
    pretrained_model.save(out_fine_tuned_model)

if __name__ == '__main__':
    model_path = '../../../data/skip_s100.txt'
    out_fine_tuned_model = os.path.join("../../../data/word2vec_fine_tuned.txt")

    corpus = load_docs_dataset('../../../data/dataset')
    train_fine_tune(model_path, corpus, out_fine_tuned_model)