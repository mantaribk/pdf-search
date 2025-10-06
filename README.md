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


https://ollama.com/library/llama3.3
```
OLLAMA_URL=http://localhost:11434
```
```bash
ollama run llama3.2:latest 
```

### run pdf

```bash
python main.py -p '~/docx.pdf'
```
