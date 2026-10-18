# Log Classification

This project classifies log messages using regex rules, a BERT-based model, and an LLM.
It provides a FastAPI endpoint that accepts a CSV file and returns the classified CSV with a `target_label` column.

## Setup

```bash
cd log_classification
python3 -m venv log_class_env
source log_class_env/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
API_KEY=your_api_key
GROQ_API_KEY=your_groq_api_key
```

## Run the API

```bash
source log_class_env/bin/activate
uvicorn main:app --reload
```

Send a CSV containing `source` and `log_message` columns:

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "X-API-Key: your_api_key" \
  -F "file=@testing/test.csv" \
  -o testing/classified_logs.csv
```

## Run from the command line

```bash
source log_class_env/bin/activate
python classify.py
```

The command reads `testing/test.csv` and writes the classified output to `test.csv`.
