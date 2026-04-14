"""
generation.py — Llama 3.1 8B Instruct Answer Generator
======================================================

Wraps the locally-hosted Llama 3.1 8B Instruct model for RAG-based
question answering. Uses HuggingFace Transformers for inference.

The prompt template follows the paper's Table 5 specification:
  System: "You are a Question Answering system..."
  User:   Context passages + Question
"""

import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


class LlamaGenerator:
    """
    Llama 3.1 8B Instruct generator for RAG question answering.

    Loads the model in bfloat16 for efficient GPU inference on RTX 5090.
    Uses the chat template format expected by Llama 3.1 Instruct models.
    """

    def __init__(
            self,
            model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
            device: str = "cuda",
            max_new_tokens: int = 100,
    ):
        """
        Initialize the Llama generator.

        Parameters
        ----------
        model_name : str
            HuggingFace model ID or local path to the model weights.
        device : str
            Device for inference ('cuda' or 'cpu').
        max_new_tokens : int
            Maximum tokens to generate per answer (paper uses 100).
        """
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"[LlamaGenerator] Loading model '{model_name}'...")
        print(f"[LlamaGenerator] Device: {device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Ensure pad token is set (Llama doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model in bfloat16 for memory-efficient inference
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Automatically places layers on GPU
            trust_remote_code=True,
        )
        self.model.eval()

        print(f"[LlamaGenerator] ✓ Model loaded successfully.")

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

        # ── Format as Llama 3.1 chat messages ─────────────────
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Apply the model's chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ── Tokenize and generate ─────────────────────────────
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,  # Llama 3.1 supports up to 128k, but we cap input
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,           # Greedy decoding for reproducibility
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # ── Extract only the generated tokens (not the prompt) ─
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return answer.strip()
