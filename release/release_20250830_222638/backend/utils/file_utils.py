import re

def clean_text(text: str) -> str:
    """Simple text cleaning utility."""
    printable = "".join(c for c in text if c.isprintable())
    return re.sub(r"\s+", " ", printable).strip()
