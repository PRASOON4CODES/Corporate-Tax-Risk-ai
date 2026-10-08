def generate_risk_score(probabilities):
    """
    Convert model probabilities into a 0-100 risk score.
    """

    low = probabilities.get("Low", 0)
    medium = probabilities.get("Medium", 0)
    high = probabilities.get("High", 0)

    score = (
        low * 20
        + medium * 60
        + high * 100
    )

    return round(score, 2)


def generate_risk_factors(data):
    """
    Generate human-readable risk indicators.
    """

    factors = []

    if data["previous_fines"] > 1000000:
        factors.append("High previous fines")

    elif data["previous_fines"] > 100000:
        factors.append("Moderate previous fines")

    if data["internal_control_score"] < 60:
        factors.append("Weak internal controls")

    elif data["internal_control_score"] < 80:
        factors.append("Moderate internal controls")

    if data["effective_tax_rate"] < 10:
        factors.append("Unusually low effective tax rate")

    if data["offshore_transactions"] >= 5:
        factors.append("High offshore transaction activity")

    if data["restatement_history"] > 0:
        factors.append("Financial restatement history")

    if data["governance_score"] < 60:
        factors.append("Weak governance score")

    if data["board_independence"] < 50:
        factors.append("Low board independence")

    if data["whistleblower_reports"] >= 4:
        factors.append("Multiple whistleblower reports")

    if data["leverage_ratio"] > 2:
        factors.append("High leverage ratio")

    if not factors:
        factors.append("No major rule-based risk indicators detected")

    return factors