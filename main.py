from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import json

from pdfminer.high_level import extract_text
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForTokenClassification

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
def process_pdf(pdf_path):
    text = extract_text(pdf_path)

    tokenizer = AutoTokenizer.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")
    model = AutoModelForTokenClassification.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")

    nlp = hf_pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=-1,   # CPU
        truncation=True,
    max_length=512 
    )
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


@app.post("/extract")
async def extract_entities(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Run pipeline
        results = process_pdf(tmp_path)

        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import logging

@app.post("/redact")
async def extract_entities(file: UploadFile = File(...)):
    logger.info(f"Received request to /extract with file: {file.filename}")

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
        logger.info(f"Extraction completed, found {len(results)} entities")

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
