import math


def temperature_scaled_softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    safe_temperature = max(temperature, 1e-6)
    shifted = [value / safe_temperature for value in logits]
    max_logit = max(shifted)
    exp_values = [math.exp(value - max_logit) for value in shifted]
    total = sum(exp_values)
    return [value / total for value in exp_values]

