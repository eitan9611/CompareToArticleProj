"""
generation.py — Llama 3.1 8B Instruct Answer Generator (via Ollama)
===================================================================

Wraps the locally-hosted Llama 3.1 8B Instruct model via Ollama's
REST API for RAG-based question answering.

The prompt template follows the paper's Table 5 specification:
  System: "You are a Question Answering system..."
  User:   Context passages + Question

NOTE: Requires Ollama running locally (default: http://localhost:11434).
      Model must be pulled first: `ollama pull llama3.1:8b`
"""

import json
import logging
from typing import List

import requests

log = logging.getLogger(__name__)


class LlamaGenerator:
    """
    Llama 3.1 8B Instruct generator for RAG question answering.

    Uses Ollama's local REST API — no HuggingFace token or manual
    quantization needed. Ollama handles model loading and memory
    management automatically.
    """

    def __init__(
            self,
            model_name: str = "llama3.1:8b",
            ollama_base_url: str = "http://localhost:11434",
            max_new_tokens: int = 100,
            **kwargs,  # Accept and ignore extra kwargs for backward compat
    ):
        """
        Initialize the Llama generator (Ollama backend).

        Parameters
        ----------
        model_name : str
            Ollama model name (e.g., "llama3.1:8b").
        ollama_base_url : str
            Base URL for the Ollama API.
        max_new_tokens : int
            Maximum tokens to generate per answer (paper uses 100).
        """
        self.model_name = model_name
        self.base_url = ollama_base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens

        print(f"[LlamaGenerator] Using Ollama backend at {self.base_url}")
        print(f"[LlamaGenerator] Model: {model_name}")

        # Verify Ollama is reachable and model is available
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            # Check if the model (or a variant) is available
            model_found = any(
                model_name in name or name.startswith(model_name.split(":")[0])
                for name in available
            )
            if not model_found:
                print(f"[LlamaGenerator] ⚠ Model '{model_name}' not found in Ollama. "
                      f"Available: {available}")
                print(f"[LlamaGenerator]   Run: ollama pull {model_name}")
            else:
                print(f"[LlamaGenerator] ✓ Model '{model_name}' is available.")
        except requests.ConnectionError:
            print(f"[LlamaGenerator] ⚠ Could not connect to Ollama at {self.base_url}. "
                  f"Make sure Ollama is running.")
        except Exception as e:
            print(f"[LlamaGenerator] ⚠ Error checking Ollama: {e}")

    def generate(
            self,
            question: str,
            context_passages: List[str],
            system_prompt: str,
    ) -> str:
        """
        Generate an answer given a question and retrieved context passages.

        Parameters
        ----------
        question : str
            The question to answer.
        context_passages : List[str]
            Retrieved passages from ChromaDB (top-K).
        system_prompt : str
            The system prompt (from paper's Table 5).

        Returns
        -------
        str
            The generated answer text.
        """

        # ── Build the context string from retrieved passages ───
        context_str = "\n\n".join(
            f"[Passage {i + 1}]: {passage}"
            for i, passage in enumerate(context_passages)
        )

        # ── Build the user message ─────────────────────────────
        user_message = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        # ── Call Ollama's chat API ─────────────────────────────
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": 0.0,  # Greedy decoding for reproducibility
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            answer = result.get("message", {}).get("content", "")
            return answer.strip()

        except requests.ConnectionError:
            log.error("Cannot connect to Ollama. Is it running?")
            return ""
        except Exception as e:
            log.error(f"Ollama generation error: {e}")
            return ""
