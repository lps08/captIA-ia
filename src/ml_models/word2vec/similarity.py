#%%
from gensim.models import KeyedVectors
import numpy as np
import re
import nltk
from unidecode import unidecode
from nltk.corpus import stopwords
import os
import requests
import zipfile
from src import constants

def load_model(model_path, url_model = constants.URL_WOR2VEC_MODEL):
    """
    Load a word2vec model from the specified path or download it from the provided URL.

    Args:
        model_path (str): The path to the word2vec model file.
        url_model (str): The URL to download the word2vec model from. Defaults to constants.URL_WOR2VEC_MODEL.

    Returns:
        gensim.models.KeyedVectors or None: The loaded word2vec model if successful, else None.

    Note:
        This function attempts to load a word2vec model from the specified path.
        If the model file doesn't exist, it downloads the model from the provided URL and saves it to the specified path.
        Once downloaded and extracted, it loads the model using gensim's KeyedVectors.

    Example:
        >>> model_path = 'path/to/word2vec_model.txt'
        >>> word2vec_model = load_model(model_path)
    """
    if not os.path.exists(model_path):
        response = requests.get(url_model, stream=True)
        if response.status_code == 200:
            print('Downloading word2vec model...')
            zip_file = f"{os.path.dirname(model_path)}/skip_s100.zip"
            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {model_path}")
            print(f"Extracting {zip_file}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(model_path))
            print(f"File extracted to {os.path.dirname(model_path)}")
            os.remove(zip_file)
        else:
            print(f"Failed to download {model_path} with status code {response.status_code}")
            return None

    return KeyedVectors.load_word2vec_format(model_path)

def preprocess_text(text, lang="portuguese"):
    """
    Preprocesses the given text by removing non-alphabetic characters, tokenizing, removing stopwords, and converting to lowercase.

    Args:
        text (str): The text to be preprocessed.
        lang (str, optional): The language of the text. Defaults to "portuguese".

    Returns:
        list: A list of preprocessed tokens.

    Note:
        This function preprocesses the given text by performing the following steps:
        1. Removes non-alphabetic characters.
        2. Tokenizes the text using NLTK's word_tokenize function.
        3. Removes stopwords specific to the specified language.
        4. Converts tokens to lowercase.

    Example:
        >>> text = "Esta é uma amostra de texto para pré-processamento."
        >>> preprocessed_text = preprocess_text(text)
    """
    text = re.sub(r'[^a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ\s]', '', text).strip().lower()
    tokens = nltk.word_tokenize(text, language=lang)
    tokens = [unidecode(t) for t in tokens if t not in stopwords.words(lang)]

    return tokens

def get_most_similar_candidate(model, targets, candidates, threshold=0.35, boost_factor=1.5):
    """
    Finds the most similar candidate among a list of candidates based on their similarity to a list of target phrases.

    Args:
        model: The word embedding model used to compute similarities.
        targets (list): A list of target phrases.
        candidates (list): A list of candidate phrases.
        threshold (float, optional): The similarity threshold for considering a candidate as similar to a target. Defaults to 0.35.
        boost_factor (float, optional): The factor by which to boost the similarity score for single-word candidates that match a target word. Defaults to 1.5.

    Returns:
        str or None: The most similar candidate phrase if its similarity score is above the threshold, otherwise None.

    Note:
        This function calculates the similarity between each target phrase and each candidate phrase using the word embedding model.
        It then aggregates the similarity scores for each candidate across all target phrases, taking into account a boost factor for single-word candidates that match a target word.
        The candidate with the highest aggregate score is returned as the most similar candidate if its score is above the threshold.

    Example:
        >>> model = Word2Vec.load("word2vec_model.txt")
        >>> targets = ["objective", "eligibility criteria"]
        >>> candidates = ["goals", "criteria", "requirements"]
        >>> threshold = 0.4
        >>> boost_factor = 2.0
        >>> get_most_similar_candidate(model, targets, candidates, threshold, boost_factor)
    """
    target_tokens = [preprocess_text(target) for target in targets]
    candidate_tokens = [preprocess_text(candidate) for candidate in candidates]
    
    similarities_for_candidates = []

    for target in target_tokens:
        similarities = []
        for candidate in candidate_tokens:
            is_single_word = len(candidate) == 1

            penalty_factor = boost_factor if is_single_word else 1

            boost = penalty_factor * (1 if is_single_word and candidate[0] in target else 0)
            similarity = model.n_similarity(target, candidate) + boost
            similarities.append(similarity)
        similarities_for_candidates.append(similarities)

    aggregate_scores = np.mean(similarities_for_candidates, axis=0)
    
    best_candidate_index = np.argmax(aggregate_scores)
    best_candidate = candidates[best_candidate_index]

    # print(best_candidate, aggregate_scores[best_candidate_index])

    if aggregate_scores[best_candidate_index] > threshold:
        del candidates[best_candidate_index]
        return best_candidate

    return None