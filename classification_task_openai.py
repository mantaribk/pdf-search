from typing import List
from openai import AsyncOpenAI
from pathlib import Path
from core.config import ollama_url

PROMPTS_DIR = Path("prompts")

labels_prompt = (PROMPTS_DIR / "labels_prompt_openai.txt").read_text(encoding="utf-8")

client = AsyncOpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")


async def get_llm_full_response(model_id: str, prompt_template: str, **kwargs) -> str:
    prompt = prompt_template.format(**kwargs)
    response = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        top_p=1.0,
        response_format={"type": "json_object"},
        stream=False,
        extra_body={
            "options": {
                "seed": 42,
                "top_k": 0,
                "mirostat": 0,
                "repeat_penalty": 1.0,
                "num_ctx": 8192,
                "num_predict": 2048,
                "stop": [],
            }
        },
    )
    return response.choices[0].message.content


async def label_page_numbers(
    page_number: int, page_text: str, numbers_json: List[dict[str, str]], model_id: str
) -> str:
    prompt_template = labels_prompt
    try:
        full_response = await get_llm_full_response(
            model_id,
            prompt_template,
            page_number=page_number,
            page_text=page_text,
            numbers_json=numbers_json,
        )
        return page_number, full_response
    except Exception as e:
        print(f"Error calling Ollama service: {e}")
        return f"Error: {str(e)}"
