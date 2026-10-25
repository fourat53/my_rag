def process_text(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped.split()) < 4:
            continue
        if stripped not in seen:
            seen.add(stripped)
            cleaned_lines.append(stripped)
    cleaned_text = " ".join(cleaned_lines)
    return cleaned_text
