from pdfminer.high_level import extract_text
import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine
# helper functions
# %%
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
        # If same entity and consecutive (index not strictly needed if you trust order)
        if token["entity"] == current["entity"] and token["start"] == current["end"]:
            # Merge the word and extend the end
            current["word"] += token["word"]
            current["end"] = token["end"]
        else:
            merged.append(current)
            current = token.copy()
    merged.append(current)
    return merged

def remove_special_characters(word):
    if word[0] == "▁":
        word = word[1:]
    return word.replace("▁", " ")

# %%

def pipeline(pdf_path):
    
    text = extract_text(pdf_path)

    tokenizer = AutoTokenizer.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")
    model = AutoModelForTokenClassification.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")

    nlp = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=-1
    )
    results = nlp(text)

    results = merge_tokens(results)
    for r in results:
        r['entity'] = map_label(r['entity'])
        r['word'] = remove_special_characters(r["word"])

    json_results = json.dumps([
        {
            "entity": r.get("entity_group", r.get("entity")),
            "score": float(r["score"]),   # Convert numpy.float32 → float
            "word": r["word"],
            "start": r.get("start"),
            "end": r.get("end")
        }
        for r in results
    ], indent=2)