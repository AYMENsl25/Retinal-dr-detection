from app.schemas.reports import ClinicalReport

DISCLAIMER = "Decision support only - not a medical diagnosis. Confirm all findings with a licensed ophthalmologist."


def enforce_clinical_guardrails(report: ClinicalReport) -> ClinicalReport:
    if "diagnosis" not in report.disclaimer.lower() and "decision support" not in report.disclaimer.lower():
        report.disclaimer = DISCLAIMER
    return report

