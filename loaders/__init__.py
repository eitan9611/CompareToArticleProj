"""
loaders — Dataset loaders for the RAG benchmarking project.

Each loader returns:
  - documents: List[langchain_core.documents.Document]  (for ingestion)
  - evaluation_set: List[dict]                          (for QA evaluation)
"""

from loaders.squad_loader import SquadLoader
from loaders.covidqa_loader import CovidQALoader
from loaders.nq_loader import NQLoader

__all__ = ["SquadLoader", "CovidQALoader", "NQLoader"]
