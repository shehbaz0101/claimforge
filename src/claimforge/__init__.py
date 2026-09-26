"""ClaimForge: live scientific claim verification."""

from claimforge.extract import extract_claims_from_abstract, extract_from_openalex_work
from claimforge.judge import judge_claim
from claimforge.models import Claim, ClaimType, Evidence, EvidenceSource, Verdict, VerdictLabel
from claimforge.retrieve import retrieve_evidence

__version__ = "0.1.0"

__all__ = [
    "Claim",
    "ClaimType",
    "Evidence",
    "EvidenceSource",
    "Verdict",
    "VerdictLabel",
    "__version__",
    "extract_claims_from_abstract",
    "extract_from_openalex_work",
    "judge_claim",
    "retrieve_evidence",
]
