import re

# =========================================================
# OCR / PDF NOISE CLEANING
# =========================================================

def remove_ocr_noise(text: str) -> str:
    """
    Remove obvious OCR garbage and meaningless long random strings.
    """

    # Remove extremely long random alphanumeric garbage
    text = re.sub(r'\b[A-Za-z0-9]{35,}\b', ' ', text)

    # Remove repeated underscores/dashes
    text = re.sub(r'[_\-]{4,}', ' ', text)

    # Remove broken page artifacts
    text = re.sub(r'\bPage\s+\d+\b', ' ', text, flags=re.IGNORECASE)

    # Remove common footer/header noise
    text = re.sub(
        r'Document Version\s*\d+(\.\d+)?',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    return text

    # =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    """
    Normalize spacing while preserving
    API symbols, JSON structure, and technical syntax.
    """

    # Normalize newlines
    text = text.replace('\r', '\n')

    # Remove excessive empty lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)

    # Trim spaces around newlines
    text = re.sub(r' *\n *', '\n', text)

    return text.strip()


# =========================================================
# MAIN CLEANING FUNCTION
# =========================================================

def clean_document(text: str) -> str:
    """
    Clean document while preserving semantic structure.
    """

    # Remove OCR garbage
    text = remove_ocr_noise(text)

    # Normalize formatting
    text = normalize_text(text)

    return text
