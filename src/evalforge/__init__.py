"""EvalForge: evaluate LLM outputs and the judges that score them."""

from evalforge.agreement import AgreementReport, ConfusionMatrix, agreement
from evalforge.models import Case, Verdict

__all__ = [
    "AgreementReport",
    "Case",
    "ConfusionMatrix",
    "Verdict",
    "agreement",
]
__version__ = "0.1.0"
