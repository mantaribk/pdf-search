from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from typing import List, Optional, Dict
from core.config import ollama_url

PROMPTS_DIR = Path("prompts")

labels_prompt = (PROMPTS_DIR / "labels_prompt.txt").read_text(encoding="utf-8")

class NumberResult(BaseModel):
    raw_value: str = Field(description="Original extracted value")
    formatted_value: int = Field(description="only number without other symbols and units")
    semantic_label: str = Field(..., description="One of predefined labels (e.g., grant_amount)")
    semantic_category: str = Field(..., description="One of predefined categories (e.g., FINANCIAL VALUES)")
    unit: Optional[str] = Field(None, description="$|%|x|units|null")
    llm_confidence: float = Field(description="0.0–1.0 confidence")

class PageResults(BaseModel):
    results: List[NumberResult]
    page: int

def get_number_labeler_chain(model_id: str):
    """Return asynchronous iterator for LLM responses with variable substitution in the prompt"""
    parser = JsonOutputParser(pydantic_object=PageResults)
    prompt = PromptTemplate(
        input_variables=["page_number", "page_text", "numbers_json"],
        template=labels_prompt,
        partial_variables={
            "format_instructions": JsonOutputParser(pydantic_object=PageResults).get_format_instructions()
        },
    )
    llm = ChatOllama(model=model_id, base_url=ollama_url, temperature=0, num_ctx=2048)
    return prompt | llm | parser


async def label_page_numbers(
    page_number: int, page_text: str, numbers_json: List[dict], model_id: str
):
    chain = get_number_labeler_chain(model_id)
    return await chain.ainvoke(
        {
            "page_number": page_number,
            "page_text": page_text,
            "numbers_json": numbers_json,
        }
    )