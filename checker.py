import re

def extract_number(text):
    matches = re.findall(r'final answer[\s\*:\-]*\$?\s*(-?[\d,]*\.?\d+)', text, re.I)
    if matches:
        return matches[-1].replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return nums[-1] if nums else None

def is_correct(answer_text, correct_answer):
    got = extract_number(answer_text)
    if got is None:
        return False
    try:
        return abs(float(got) - float(correct_answer)) < 1e-6
    except ValueError:
        return False