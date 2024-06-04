#%%
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
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

def get_prompt_role(parser, tokenizer_name=constants.LOCAL_LLM_MODEL_NAME):
    """
    Creates a prompt template for a language model to follow specific roles and instructions.

    This function defines a prompt template for an assistant role that answers questions based on provided context. The assistant should only answer what is asked and avoid making up answers. The function uses a specified tokenizer to apply a chat template to the messages.

    Args:
        parser: An object that provides format instructions for the model's responses.
        tokenizer_name (str, optional): The name of the tokenizer to use for applying the chat template. Defaults to `constants.LOCAL_LLM_MODEL_NAME`.

    Returns:
        PromptTemplate: A template configured with the given messages and format instructions, ready for use with a language model.

    Example:
        >>> parser = SomeParser()
        >>> prompt_template = get_prompt_role(parser)
    """
    messages = [
        {
            "role": "system",
            "content": "You are a assistant and answer what is asked. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know returning 'Não encontrado', don't try to make up an answer! Answer only what is asked, please! Do not include any comments or notes in the outputed answer!",
        },
        {
            "role": "user",
            "content": 'Context: \n{context}\nBased on the information of the context, answer this question: {question}.\nFollow the requirements below:\n{format_instructions}',
        },
    ]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    prompt_template_instructions = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    return PromptTemplate(
        template=prompt_template_instructions, 
        input_variables=['context', 'question'], 
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

def qa_llm(query, llm, retriever, parser=None, use_role=False, chain_type = "stuff"):
    """
    Executes a question-answering task using a language model (LLM) and a retriever.

    This function takes a query and utilizes a language model along with a retriever to extract relevant information from documents. It supports two types of prompts: a role-based prompt or a standard prompt, which can be customized with a parser.

    Args:
        query (str): The question or query to be answered by the language model.
        llm: The language model to use for generating the answer.
        retriever: The retriever used to fetch relevant documents or passages.
        parser (optional): An object that provides format instructions for the model's responses. Defaults to None.
        use_role (bool, optional): Whether to use a role-based prompt. Defaults to False.
        chain_type (str, optional): The type of chain to use for the retrieval-based QA. Defaults to "stuff".

    Returns:
        str: The result of the question-answering task, extracted from the response dictionary.

    Example:
        >>> query = "What is the objective of the project described in the document?"
        >>> llm = get_language_model()
        >>> retriever = create_retriever(documents, embeddings)
        >>> parser = SomeParser()
        >>> answer = qa_llm(query, llm, retriever, parser, use_role=True)
        >>> print(answer)
    """
    if use_role:
        prompt = get_prompt_role(parser)
    else:
        prompt = get_prompt(parser) if parser else None
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever, chain_type_kwargs={"prompt" : prompt})
    res = chain.invoke(query)
    return res['result']