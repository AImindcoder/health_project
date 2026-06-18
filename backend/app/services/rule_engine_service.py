from hackthathon import compute_rule_score


def evaluate(labs: dict, symptoms: str = "") -> dict:
    return compute_rule_score(labs, symptoms)
