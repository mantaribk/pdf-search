## Quick Start

### Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
poetry install --no-root
```

Download Ollama:
https://ollama.com/download/mac

.env file:
```
OLLAMA_URL=http://localhost:11434
```
### Load Qwen3 - the latest generation of large language models
```bash
ollama run qwen3:8b
```

### If need to use only in-house model:
Using Ministral2.5 model (included) from here:
(https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/blob/main/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf)
Run command
```bash
ollama create my-own-model -f Modelfile
```

### run pdf
(default output file extraction_results.json)
```bash
python main.py -p '~/docx.pdf'
```
or
```bash
python main.py -p '~/docx.pdf' --output '~/extraction_results.json'
```