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

Using Ministral2.5 model 
(https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/blob/main/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf)
Download Ollama:
https://ollama.com/download/mac

.env file:
```
OLLAMA_URL=http://localhost:11434
```

```bash
ollama create my-own-model -f Modelfile
```

### run pdf

```bash
python main.py -p '~/docx.pdf'
```



