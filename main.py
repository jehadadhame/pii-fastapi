from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import json

from pdfminer.high_level import extract_text
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForTokenClassification
import logging
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Helper functions
# -------------------------
def map_label(label):
    if label == "O":
        return label
    return label.split("-")[-1] 

def merge_tokens(tokens):
    merged = []
    if not tokens:
        return merged
    
    current = tokens[0].copy()
    for token in tokens[1:]:
        if token["entity"] == current["entity"] and token["start"] == current["end"]:
            current["word"] += token["word"]
            current["end"] = token["end"]
        else:
            merged.append(current)
            current = token.copy()
    merged.append(current)
    return merged

def remove_special_characters(word):
    if word and word[0] == "▁":
        word = word[1:]
    return word.replace("▁", " ")


# -------------------------
# Core pipeline function
# -------------------------

MODEL_NAME = "iiiorg/piiranha-v1-detect-personal-information"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
nlp = hf_pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,
)

def process_pdf(pdf_path):
    text = extract_text(pdf_path)

    results = nlp(text)

    results = merge_tokens(results)
    for r in results:
        r['entity'] = map_label(r['entity'])
        r['word'] = remove_special_characters(r["word"])

    json_results = [
        {
            "entity": r.get("entity_group", r.get("entity")),
            "score": float(r["score"]),
            "word": r["word"],
            "start": r.get("start"),
            "end": r.get("end")
        }
        for r in results
    ]

    return json_results

def test_pipline():


    text = '''
    John Doe's email is
    johndoe@gmail.com and his phone number is +1-202-555-0173.
    His credit card number is 4111 1111 1111 1111 and his SSN is 123-45-6789.
    He lives at 123 Main St, Springfield, IL 62701.
    '''
    results = nlp(text)

    results = merge_tokens(results)
    for r in results:
        r['entity'] = map_label(r['entity'])
        r['word'] = remove_special_characters(r["word"])

    json_results = [
        {
            "entity": r.get("entity_group", r.get("entity")),
            "score": float(r["score"]),
            "word": r["word"],
            "start": r.get("start"),
            "end": r.get("end")
        }
        for r in results
    ]

    return json_results


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI()


@app.get("/test")
async def test():
    logger.info(f"Received request to /test ")
    
    results = test_pipline()
    return JSONResponse(content=results)



@app.post("/redact")
async def extract_entities(file: UploadFile = File(...)):
    logger.info(f"Received request to /redact with file: {file.filename}")
    start_time = time.time()

    if not file.filename.endswith(".pdf"):
        logger.warning(f"Rejected file {file.filename} (not a PDF)")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file_bytes = await file.read()
            tmp.write(file_bytes)
            tmp_path = tmp.name
        logger.info(f"Saved uploaded PDF to temp file: {tmp_path} (size: {len(file_bytes)} bytes)")

        # Run pipeline
        logger.info(f"Starting entity extraction for {file.filename}")
        results = process_pdf(tmp_path)
        elapsed = time.time() - start_time
        logger.info(f"Extraction completed in {elapsed:.2f}s, found {len(results)} entities")

        logger.info(f"Extraction completed, found {len(results)} entities")

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
