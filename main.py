import tempfile
import os
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv

from fastapi import (
    FastAPI, UploadFile, HTTPException, Security, Depends,
    BackgroundTasks
)
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader

from classify import classify

# dataset_path = Path(__file__).parent / "dataset/training_logs.csv"
# df = pd.read_csv(dataset_path)

load_dotenv()

app = FastAPI()

api_key = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.post("/classify", dependencies=[Depends(verify_api_key)])
async def classify_logs(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a CSV file."
        )

    try:
        df = pd.read_csv(file.file)

        df['target_label'] = classify(list(zip(df['source'], df['log_message'])))

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.close()
        df.to_csv(tmp.name, index=False)

        background_tasks.add_task(os.remove, tmp.name)

        return FileResponse(
            tmp.name, media_type="text/csv", filename="classified_logs.csv"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

    finally:
        file.file.close()