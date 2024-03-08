#%%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from src.ml_models.llm.retrieval_qa_llm import qa_llm
from src.ml_models.llm.preprocessing import load_doc, create_retriever
from src import constants
import langchain
langchain.debug = True

def get_gemma_model(
        name = constants.GEMMA_MODEL_NAME,
        padding=True, 
        truncation=True, 
        max_length=512, 
        repetition_penalty=1.1,
        max_new_tokens=512,
        return_full_text=True,
        temperature=0.1,
        model_cache_dir="data/",
        task_pipline_name="text-generation"
    ):
    """
    Get an instance of the GEMMA model for text generation.

    Args:
        name (str, optional): The name or path of the GEMMA model. Defaults to constants.GEMMA_MODEL_NAME.
        padding (bool, optional): Whether to pad sequences to the maximum length. Defaults to True.
        truncation (bool, optional): Whether to truncate sequences to the maximum length. Defaults to True.
        max_length (int, optional): The maximum sequence length. Defaults to 512.
        repetition_penalty (float, optional): The repetition penalty applied during generation. Defaults to 1.1.
        max_new_tokens (int, optional): The maximum number of new tokens generated. Defaults to 512.
        return_full_text (bool, optional): Whether to return the full generated text. Defaults to True.
        temperature (float, optional): The temperature for generation sampling. Defaults to 0.1.
        model_cache_dir (str, optional): The directory to cache the model and tokenizer. Defaults to "data/".
        task_pipeline_name (str, optional): The name of the task pipeline. Defaults to "text-generation".

    Returns:
        HuggingFacePipeline: An instance of the GEMMA model for text generation.

    Note:
        This function returns an instance of the GEMMA model for text generation.
        It allows customization of various generation parameters such as padding, truncation, repetition penalty, etc.
        The model and tokenizer are loaded using AutoModelForCausalLM and AutoTokenizer from the Hugging Face library.
        The returned pipeline instance is configured for text generation with the specified parameters.

    Example:
        >>> gemma_model = get_gemma_model()
        >>> generated_text = gemma_model.invoke("What are the latest trends in AI?")
        >>> print(generated_text)
        "The latest trends in AI include..."
    """
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir=model_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(name, padding=padding, truncation=truncation, max_length=max_length, cache_dir=model_cache_dir)

    model_pipeline = pipeline(
        task=task_pipline_name,
        model=model,
        tokenizer=tokenizer,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        return_full_text=return_full_text,
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs={
            "temperature": temperature
        },
    )
    return llm