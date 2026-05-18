# =========================================================
# CONTENT CLASSIFICATION
# =========================================================

def classify_content(text: str) -> str:
    """
    Detect technical/API-related content types.
    """

    text_lower = text.lower()

    # API endpoint detection
    if (
        "/api/" in text_lower or
        "endpoint" in text_lower or
        "http://" in text_lower or
        "https://" in text_lower
    ):
        return "api_endpoint"

    # JSON examples
    if (
        "{" in text and
        "}" in text and
        ":" in text
    ):
        return "json_example"

    # Parameter tables
    if any(
        kw in text_lower
        for kw in [
            "parameter",
            "sample value",
            "description",
            "type"
        ]
    ):
        return "parameter_definition"

    # Authentication docs
    if any(
        kw in text_lower
        for kw in [
            "token",
            "authorization",
            "bearer",
            "api key"
        ]
    ):
        return "authentication"

    # Structural sections
    if any(
        kw in text_lower
        for kw in [
            "chapter",
            "section",
            "introduction"
        ]
    ):
        return "structural"

    return "prose"


# =========================================================
# SECTION TITLE EXTRACTION
# =========================================================

def extract_section_title(text: str) -> str:
    """
    Try extracting heading/section title from chunk.
    """

    lines = text.split('\n')

    for line in lines[:10]:

        cleaned = line.strip()

        # Skip very short lines
        if len(cleaned) < 5:
            continue

        # Skip overly long lines
        if len(cleaned) > 120:
            continue

        # Detect likely heading
        if (
            cleaned.lower().startswith("section") or
            cleaned.lower().startswith("chapter") or
            "api" in cleaned.lower() or
            cleaned.endswith(":")
        ):
            return cleaned

    return "Unknown Section"


# =========================================================
# SEMANTIC FEATURE DETECTION
# =========================================================

def detect_semantic_features(text: str) -> dict:

    text_lower = text.lower()

    return {
        "contains_json": (
            "{" in text and
            "}" in text
        ),

        "contains_url": (
            "http://" in text_lower or
            "https://" in text_lower
        ),

        "contains_api": (
            "/api/" in text_lower or
            "api" in text_lower
        ),

        "contains_table_like_data": (
            "sample value" in text_lower or
            "description" in text_lower
        )
    }
