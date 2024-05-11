#%%
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from src import constants

def get_google_embeddings(model_name=constants.GOOGLE_EMBEDDINGS_MODEL_NAME):
    """
    Get embeddings from a Google Generative AI model.

    Args:
        model_name (str, optional): The name or path of the Google Generative AI model. Defaults to constants.GOOGLE_EMBEDDINGS_MODEL_NAME.

    Returns:
        GoogleGenerativeAIEmbeddings: An instance of the embeddings model.

    Note:
        This function returns an instance of GoogleGenerativeAIEmbeddings, which provides embeddings from a Google Generative AI model.
        The model_name parameter specifies the name or path of the Google Generative AI model to be used for generating embeddings.

    Example:
        >>> embeddings_model = get_google_embeddings()
        >>> embeddings = embeddings_model.invoke("Hello, world!")
    """
    return GoogleGenerativeAIEmbeddings(model=model_name)

def get_huggingface_embeddings(
        model_name = constants.HUGGINGFACE_EMBEDDINGS_MODEL_NAME, 
        device="cpu", 
        normalize_embeddings=True
    ):
    """
    Get embeddings from a HuggingFace model.

    Args:
        model_name (str, optional): The name or path of the Hugging Face model. Defaults to constants.HUGGINGFACE_EMBEDDINGS_MODEL_NAME.
        device (str, optional): The device to run the model on. Defaults to "cpu".
        normalize_embeddings (bool, optional): Whether to normalize the embeddings. Defaults to True.

    Returns:
        HuggingFaceEmbeddings: An instance of the embeddings model.

    Note:
        This function returns an instance of HuggingFaceEmbeddings, which provides embeddings from a Hugging Face model.
        The model_name parameter specifies the name or path of the Hugging Face model to be used for generating embeddings.
        Additional parameters control the behavior of the embeddings, such as the device to run the model on and whether to normalize the embeddings.

    Example:
        >>> embeddings_model = get_huggingface_embeddings()
        >>> embeddings = embeddings_model.generate_embeddings("Hello, world!")
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize_embeddings},
    )

def get_prompt(parser):
    """
    Get a prompt template for generating questions based on a parser.

    Args:
        parser: The parser object used for generating format instructions.

    Returns:
        PromptTemplate: A prompt template for generating questions.

    Note:
        This function generates a prompt template for generating questions based on a parser.
        It utilizes the provided parser object to generate format instructions for the prompt template.
        The prompt template includes placeholders for context, format instructions, and questions.
        It provides instructions on how to format the answers and what to do if a question cannot be answered.

    Example:
        >>> my_parser = MyParser()
        >>> prompt = get_prompt(my_parser)
    """
    prompt_template_instructions = """
    Você é um assistente e responde o que é perguntado. Use as seguintes partes do contexto para responder à pergunta no final. Se não souber a resposta, basta dizer que não sabe retornando 'Não encontrado', não tente inventar uma resposta! Responda apenas o que for perguntado, por favor! Não inclua comentários ou notas na resposta gerada!: 
    {context}
    
    {format_instructions}

    Pergunta: {question}
    Answer:
    """
    return PromptTemplate(
        template=prompt_template_instructions, 
        input_variables=['context', 'question'], 
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

def qa_llm(query, llm, retriever, parser=None, chain_type = "stuff"):
    """
    Perform question answering using a language model and a retriever.

    Args:
        query (str): The question/query to be answered.
        llm: The language model used for question answering.
        retriever: The retriever used for retrieving relevant documents.
        parser: The parser object used for parsing the answer. Defaults to None.
        chain_type (str, optional): The type of retrieval QA chain to use. Defaults to "stuff".

    Returns:
        dict or str: The result of the question answering process.

    Note:
        This function performs question answering using a language model and a retriever.
        It constructs a retrieval QA chain based on the provided language model, retriever, and optional parser.
        The chain is invoked with the provided query to retrieve the answer.
        If a parser is provided, the result is parsed using the parser object.
        The final result is returned as a string.

    Example:
        >>> question = "What is the capital of Brazil?"
        >>> language_model = get_language_model()
        >>> document_retriever = get_document_retriever()
        >>> answer = qa_llm(question, language_model, document_retriever)
    """
    prompt = get_prompt(parser) if parser else None
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever, chain_type_kwargs={"prompt" : prompt})
    res = chain.invoke(query)
    return res['result']