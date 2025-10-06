import argparse
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import pdfplumber
import re
import json
from classification_task import label_page_numbers
from tqdm.asyncio import tqdm_asyncio

def enhanced_load_document(file):
    pages, coordinate_numbers = extract_numbers_with_coordinates(file)
    return {"pages": pages, "coordinate_numbers": coordinate_numbers}


def extract_numbers_with_coordinates(pdf_path):
    extractions = []

    with pdfplumber.open(pdf_path) as pdf:
        pages = {}
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            full_text = page.extract_text()
            pages[page_num] = full_text

            for word in words:
                if re.search(r"\d", word["text"]):
                    re_label = classify_number(word["text"], full_text)
                    if re_label == "list_index":
                        continue
                    extractions.append(
                        {
                            "value": word["text"],
                            # 'coordinates': {
                            #     'x0': word['x0'], 'y0': word['top'],
                            #     'x1': word['x1'], 'y1': word['bottom']
                            # },
                            "page_number": page_num,
                            "context": get_context(full_text, word["text"]),
                            "syntax_label": classify_number(word["text"], full_text),
                        }
                    )

    return pages, extractions


def get_context(full_text: str, target: str, window: int = 60) -> str:
    idx = full_text.find(target)
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(full_text), idx + len(target) + window)
    snippet = full_text[start:end].replace("\n", " ")
    target_end = idx + len(target)
    rel = target_end - start
    if 0 <= rel < len(snippet):
        after = snippet[rel:]
        m = re.match(r"\s+\d+\.?(?!\s*[A-Za-z])", after)
        if m:
            snippet = snippet[:rel] + " " + after[m.end() :]
    snippet = re.sub(r"\s*\(?\d+\)?\.?\s*$", "", snippet)
    return snippet.strip()


def classify_number(text: str, context: str) -> str:
    """
    Syntax classification including UUIDs and code sections.
    Categories: list_index, performance_multiplier, monetary, percentage,
                date, granted_units, time_period, address_number,
                uuid, code_section, unknown.
    """
    ADDRESS_TERMS = re.compile(
        r"\b(street|st\.?|ave|avenue|road|rd\.?|suite|ste\.?|blvd|boulevard|lane|ln\.?|drive|dr\.?|court|ct\.?)\b",
        re.IGNORECASE,
    )
    UUID_PATTERN = re.compile(
        r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    )
    SECTION_REF_PATTERN = re.compile(r"\u00a7?\d+\(?[a-z]?\)?", re.IGNORECASE)

    t = text.strip()
    ctx = context
    if re.fullmatch(r"\d+\.", t) or re.fullmatch(r"\(\d+\)", t):
        return "list_index"

    if UUID_PATTERN.fullmatch(t):
        return "uuid"

    if re.fullmatch(r"\d+(\.\d+)?x", t.lower()):
        return "performance_multiplier"

    if t.startswith("$") or re.search(r"\b(dollars?|usd)\b", ctx, re.IGNORECASE):
        return "monetary"

    if "%" in t:
        return "percentage"

    if ADDRESS_TERMS.search(ctx) and re.fullmatch(r"\d+", t):
        return "address_number"

    if (
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)
        or re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", t)
        or re.search(r"[A-Za-z]{3,9} \d{1,2}, \d{4}", ctx)
    ):
        return "date"

    if SECTION_REF_PATTERN.fullmatch(t) or SECTION_REF_PATTERN.search(ctx):
        return "code_section"

    if re.search(r"\b(units|shares|granted)\b", ctx, re.IGNORECASE):
        return "granted_units"

    if re.search(r"\b(year|month|anniversary|vest)\b", ctx, re.IGNORECASE):
        return "time_period"

    if (
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)
        or re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", t)
        or re.search(r"[A-Za-z]{3,9} \d{1,2}, \d{4}", ctx)
    ):
        return "date"

    return "unknown"

async def process_page(semaphore, page_num, page_text, records, model_id):
    async with semaphore:
        return await label_page_numbers(
            page_number=page_num,
            page_text=page_text,
            numbers_json=records,
            model_id=model_id
        )

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("--output", default="extraction_results.json")
    args = parser.parse_args()

    print(f"Processing: {args.path}")
    results = enhanced_load_document(args.path)

    grouped = {}
    for rec in results["coordinate_numbers"]:
        grouped.setdefault(rec["page_number"], []).append(rec)

    semaphore = asyncio.Semaphore(5)
    tasks = [
        process_page(semaphore, page_num, results["pages"][page_num], records, "llama3.2:latest")
        for page_num, records in grouped.items()
    ]

    page_results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Labeling pages"):
        pr = await coro
        page_results.append(pr)
        print(json.dumps(pr, indent=2))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
