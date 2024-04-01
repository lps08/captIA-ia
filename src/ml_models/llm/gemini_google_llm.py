#%%
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.output_parsers import PydanticOutputParser
from src.ml_models.llm.base_models.edital_model import Edital
from src import constants

def disable_safety_settings():
    """
    Disable safety settings for various harm categories.

    Returns:
        dict: A dictionary mapping harm categories to harm block thresholds set to BLOCK_NONE.

    Note:
        This function returns a dictionary specifying harm categories with their corresponding harm block thresholds set to BLOCK_NONE.
        Disabling safety settings for these categories allows content related to these categories to pass through without blocking.

    Example:
        >>> settings = disable_safety_settings()
        >>> print(settings)
        {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
        }
    """
    return {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }

def get_parser():
    """
    Get an instance of PydanticOutputParser initialized with a Pydantic object.

    Returns:
        PydanticOutputParser: An instance of PydanticOutputParser initialized with a Pydantic object.

    Note:
        This function returns an instance of PydanticOutputParser initialized with a specific Pydantic object (e.g., Edital).
        The returned PydanticOutputParser instance can be used to parse and validate data according to the Pydantic object's schema.

    Example:
        >>> parser = get_parse()
        >>> parsed_data = parser.parse(raw_data)
    """
    return PydanticOutputParser(pydantic_object=Edital)

def get_gemini_model(
        name=constants.GOOGLE_GEMINI_MODEL_NAME, 
        temperature=0.3,
        top_p=0.6, 
        top_k=1,
        convert_system_message_to_human=True, 
        disable_safety=True
    ):
    """
    Get an instance of the Google Gemini generative AI model with specified settings.

    Args:
        name (str, optional): The name of the Google Gemini model. Defaults to constants.GOOGLE_GEMINI_MODEL_NAME.
        temperature (float, optional): The temperature parameter controlling the randomness of responses. Defaults to 0.2.
        convert_system_message_to_human (bool, optional): Whether to convert system messages to human-readable format. Defaults to True.
        disable_safety (bool, optional): Whether to disable safety settings for harmful content. Defaults to True.

    Returns:
        ChatGoogleGenerativeAI: An instance of the Google Gemini generative AI model.

    Note:
        This function returns an instance of the Google Gemini generative AI model with specified settings.
        It allows customization of model name, temperature, conversion of system messages, and disabling safety settings.

    Example:
        >>> gemini_model = get_gemini_model()
        >>> response = gemini_model.invoke("Hello!")
        >>> print(response)
        "Hi there!"
    """
    llm = ChatGoogleGenerativeAI(
        model=name, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k,
        convert_system_message_to_human=convert_system_message_to_human, 
        safety_settings=disable_safety_settings() if disable_safety else None
    )
    return llm