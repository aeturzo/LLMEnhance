import os
import csv
import json
from typing import Any

class Document:
    """Represents a parsed document ready for indexing."""
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

def process_file(upload_file: Any) -> Document:
    """Parse uploaded file and return Document."""
    filename = upload_file.filename
    _, ext = os.path.splitext(filename.lower())
    content = upload_file.file.read()
    text_data = ""

    if ext == ".txt":
        text_data = content.decode("utf-8", errors="ignore")
    elif ext == ".csv":
        decoded = content.decode("utf-8", errors="ignore").splitlines()
        reader = csv.reader(decoded)
        next(reader, None)
        for row in reader:
            text_data += " ".join(row) + "\n"
    elif ext == ".json":
        decoded = content.decode("utf-8", errors="ignore")
        try:
            obj = json.loads(decoded)
            text_data = json.dumps(obj, indent=2)
        except json.JSONDecodeError:
            text_data = decoded
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return Document(name=filename, content=text_data)
