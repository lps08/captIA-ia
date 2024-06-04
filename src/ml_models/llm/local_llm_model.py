#%%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from optimum.intel import OVModelForCausalLM
from langchain import HuggingFacePipeline
from transformers import pipeline
from langchain_community.llms import HuggingFaceEndpoint
from src.ml_models.llm.retrieval_qa_llm import qa_llm
from src.ml_models.llm.preprocessing import load_doc, create_retriever
from src import constants
import langchain
langchain.debug = True

def get_local_llm_model(
    name = constants.LOCAL_LLM_MODEL_NAME,
    use_hf_inference=True,
    padding=True, 
    truncation=True,
    repetition_penalty=1.05,
    max_new_tokens=1024,
    return_full_text=True,
    temperature=0.6,
    top_p=0.6,
    do_sample=True,
    model_cache_dir="data/",
    model_offload_folder="./data/.offload",
    task_pipline_name="text-generation"
):
    """
    """
    if use_hf_inference:
        llm = HuggingFaceEndpoint(
            repo_id=name, 
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        model = OVModelForCausalLM.from_pretrained(name, export=True, cache_dir=model_cache_dir, offload_folder=model_offload_folder)
        tokenizer = AutoTokenizer.from_pretrained(name, padding=padding, truncation=truncation, max_length=max_length, cache_dir=model_cache_dir)

        pipe = pipeline(
            task_pipline_name,
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            return_full_text=return_full_text,
            top_p=top_p,
        )

        llm = HuggingFacePipeline(
            pipeline=pipe,
        )

    return llm