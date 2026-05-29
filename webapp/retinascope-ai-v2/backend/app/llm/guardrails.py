DISCLAIMER = "Decision support only. Not a medical diagnosis."


def enforce_disclaimer(text: str) -> str:
    if "not a medical diagnosis" in text.lower():
        return text
    return f"{text}\n\n{DISCLAIMER}"
