from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_debug, set_verbose

set_debug(True)
set_verbose(True)

load_dotenv()


def get_available_models() -> List[str]:
    """Return a list of available LLM models"""
    # You can customize this list based on the models you want to support
    return ["gemini-2.0-flash-lite", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-sonnet"]


def run_llm_query(model: str, prompt_template: str, prompt_vars: Dict[str, Any]) -> str:
    """Run a query against the specified LLM using the template and variables"""

    # Create prompt from template
    template = ChatPromptTemplate.from_template(prompt_template)
    formatted_prompt = template.format_messages(**prompt_vars)

    # Setup the appropriate model
    if "gpt" in model.lower():
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model_name=model, temperature=0.7)
    elif "gemini" in model.lower():
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
    elif "claude" in model.lower():
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, temperature=0.7)
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Execute the query
    response = llm.invoke(formatted_prompt)

    return response.content


def stream_llm_query(model: str, prompt_template: str, prompt_vars: Dict[str, Any]):
    """Stream a query response from the LLM for real-time display"""

    # Create prompt from template
    template = ChatPromptTemplate.from_template(prompt_template)
    formatted_prompt = template.format_messages(**prompt_vars)

    # Setup the appropriate model with streaming
    if "gpt" in model.lower():
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model_name=model, temperature=0.7, streaming=True)
    elif "gemini" in model.lower():
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, temperature=0.7, streaming=True)
    elif "claude" in model.lower():
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, temperature=0.7, streaming=True)
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Return the streaming response
    for chunk in llm.stream(formatted_prompt):
        yield chunk.content
