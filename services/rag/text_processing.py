def is_informative(line: str) -> bool:
    return any(character.isalnum() for character in line)


def process_text(text: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line or not is_informative(line) or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)
