#%%
import torch

def predict(model, tokenizer, document_text, return_tensors='pt', max_length=128, padding=True, truncation=True):
    """
    Predict the class probabilities for a given document text using the provided model and tokenizer.

    Args:
        model: The sequence classification model for prediction.
        tokenizer: The tokenizer used for tokenization.
        document_text (str): The text of the document to predict the class probabilities for.
        return_tensors (str, optional): The format of the tensors to be returned. Defaults to 'pt' (PyTorch tensors).
        max_length (int, optional): The maximum sequence length for tokenization. Defaults to 128.
        padding (bool, optional): Whether to pad the sequences to the maximum length. Defaults to True.
        truncation (bool, optional): Whether to truncate the sequences to the maximum length. Defaults to True.

    Returns:
        tuple: A tuple containing the predicted class and the class probabilities.

    Note:
        This function predicts the class probabilities for a given document text using the provided model and tokenizer.
        It tokenizes the document text, generates predictions using the model, and calculates class probabilities.
        The return_tensors parameter specifies the format of the tensors to be returned (e.g., 'pt' for PyTorch tensors).
        The max_length parameter controls the maximum sequence length for tokenization.
        Padding and truncation options determine whether to pad or truncate sequences to the maximum length.

    Example:
        >>> model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> document_text = "This is a sample document."
        >>> predicted_class, class_probabilities = predict(model, tokenizer, document_text)
    """
    inputs = tokenizer(document_text, return_tensors=return_tensors, max_length=max_length, padding=padding, truncation=truncation)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities.tolist()
