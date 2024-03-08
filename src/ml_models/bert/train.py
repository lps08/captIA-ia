#%%
import sys
sys.path.append('../../../')
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import torch
from pdfminer.high_level import extract_text
from torch.utils.data import Dataset, random_split
import os
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class PDFDataset(Dataset):
    def __init__(self, root_dir, tokenizer, max_length=128):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        edital_files = [os.path.join(root_dir, 'editais', filename) for filename in os.listdir(os.path.join(root_dir, 'editais'))]
        no_edital_files = [os.path.join(root_dir, 'no-editais', filename) for filename in os.listdir(os.path.join(root_dir, 'no-editais'))]

        self.files = edital_files + no_edital_files
        self.labels = [1] * len(edital_files) + [0] * len(no_edital_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        text = extract_text(file_path)

        encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding=True, truncation=True)

        input_ids = F.pad(encoding["input_ids"], (0, self.max_length - encoding["input_ids"].size(1)))
        attention_mask = F.pad(encoding["attention_mask"], (0, self.max_length - encoding["attention_mask"].size(1)))

        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "label": label
        }
    
def load_model(path, num_labels):
    """
    Load a pre-trained BERT model and tokenizer from the specified path.

    Args:
        path (str): The path to the directory containing the pre-trained BERT model and tokenizer.
        num_labels (int): The number of labels for sequence classification.

    Returns:
        tuple: A tuple containing the loaded BERT model and tokenizer.

    Note:
        This function loads a pre-trained BERT model and tokenizer from the specified path.
        The 'path' directory should contain the model weights and configuration files.
        The 'num_labels' parameter is used to configure the tokenizer for sequence classification tasks.

    Example:
        >>> model_path = '/path/to/pretrained_model/'
        >>> num_labels = 2
        >>> model, tokenizer = load_model(model_path, num_labels)
        >>> # Use the loaded model and tokenizer for sequence classification
    """
    model = BertForSequenceClassification.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained(path, num_labels=num_labels)

    return model, tokenizer

def dataset_loader(dataset_dir, tokenizer, max_length, batch_size, train_ratio=None, shuffle=True):
    """
    Load PDF dataset and create DataLoader objects for training and testing.

    Args:
        dataset_dir (str): The directory containing the PDF dataset.
        tokenizer: The tokenizer to use for tokenizing PDF contents.
        max_length (int): The maximum sequence length for tokenization.
        batch_size (int): The batch size for DataLoader objects.
        train_ratio (float, optional): The ratio of the dataset to use for training. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the dataset before creating DataLoader objects. Defaults to True.

    Returns:
        DataLoader or tuple of DataLoaders: DataLoader object for the dataset or tuple of DataLoaders for training and testing datasets.

    Note:
        This function loads a PDF dataset from the specified directory and creates a PDFDataset object.
        If train_ratio is provided, it splits the dataset into training and testing sets.
        It then creates DataLoader objects for the dataset(s) with the specified batch size and shuffling option.

    Example:
        >>> dataset_dir = '/path/to/dataset'
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> max_length = 512
        >>> batch_size = 32
        >>> train_ratio = 0.8
        >>> train_loader, test_loader = dataset_loader(dataset_dir, tokenizer, max_length, batch_size, train_ratio=train_ratio)
    """
    pdf_dataset = PDFDataset(dataset_dir, tokenizer, max_length=max_length)

    if train_ratio:
        dataset_size = len(pdf_dataset)
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(pdf_dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    return DataLoader(pdf_dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(model, tokenizer, train_dataloader, optimizer, epochs, out_path):
    """
    Train a sequence classification model using the provided DataLoader.

    Args:
        model: The sequence classification model to train.
        tokenizer: The tokenizer used for tokenization.
        train_dataloader: The DataLoader containing training data.
        optimizer: The optimizer used for training.
        epochs (int): The number of training epochs.
        out_path (str): The directory path to save the trained model and tokenizer.

    Returns:
        None

    Note:
        This function trains a sequence classification model using the provided DataLoader.
        It iterates over the DataLoader for the specified number of epochs, computing loss and optimizing parameters.
        After training, if out_path is provided, it saves the trained model and tokenizer to the specified directory.

    Example:
        >>> model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> optimizer = AdamW(model.parameters(), lr=2e-5)
        >>> train_dataloader = DataLoader(...)
        >>> epochs = 5
        >>> out_path = '/path/to/save/model'
        >>> train_model(model, tokenizer, train_dataloader, optimizer, epochs, out_path)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    if out_path:
        model.save_pretrained(out_path)
        tokenizer.save_pretrained(out_path)

def evaluate(model, test_dataloader):
    """
    Evaluate a sequence classification model using the provided DataLoader.

    Args:
        model: The sequence classification model to evaluate.
        test_dataloader: The DataLoader containing the test data.

    Returns:
        None

    Note:
        This function evaluates a sequence classification model using the provided DataLoader.
        It iterates over the DataLoader, computes predictions, and evaluates the model's performance.
        It prints accuracy, classification report, and confusion matrix based on the predictions.

    Example:
        >>> model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        >>> test_dataloader = DataLoader(...)
        >>> evaluate(model, test_dataloader)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    true_labels = []
    predicted_labels = []

    for batch in test_dataloader:
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            logits = model(inputs, attention_mask=attention_mask).logits
            predictions = torch.argmax(logits, dim=1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)

    classification_rep = classification_report(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    model_name = 'deepset/roberta-base-squad2'
    num_labels = 2
    learning_rate = 1e-5
    num_epochs = 3
    max_lenght = 128
    batch_size = 16
    train_ratio = 0.2
    model_out_path = '../../../data/bert_edital_classifier'

    model, tokenizer = load_model(path=model_name, num_labels=num_labels)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_dataloader, test_dataloader = dataset_loader(
        dataset_dir='../../../data/dataset/', 
        tokenizer=tokenizer,
        max_length=max_lenght,
        batch_size=batch_size,
        train_ratio=train_ratio
    )
    train_model(model, tokenizer, train_dataloader, optimizer, num_epochs, model_out_path)

    model = load_model(
        path=model_out_path,
        num_labels=num_labels
    )
    evaluate(model=model, test_dataloader=test_dataloader)