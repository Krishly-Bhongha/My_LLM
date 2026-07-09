def extract_message(line: str) -> str:
    """
    Krish Garg: Hello
    ->
    Hello
    """

    if ":" not in line:
        return ""

    return line.split(":", 1)[1].strip()