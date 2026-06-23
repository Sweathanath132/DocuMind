"""Local LLM wrapper.

We use google/flan-t5-base, an instruction-tuned seq2seq model that runs on CPU
with no API key. flan-t5 is a "Small Language Model" (SLM) by today's standards
(~250M params) — ideal for low-latency local inference, exactly the SLM angle on
the resume. Swap settings.llm_model to flan-t5-large for higher quality, or wire
in an API client here if you later get a key.

We load the tokenizer + model directly (instead of the high-level `pipeline`
task string) so this works across transformers versions, including 5.x where the
"text2text-generation" task alias was removed.
"""
from __future__ import annotations
from .config import settings

_tok = None
_model = None


def _load():
    global _tok, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _tok = AutoTokenizer.from_pretrained(settings.llm_model)
        _model = AutoModelForSeq2SeqLM.from_pretrained(settings.llm_model)
    return _tok, _model


def generate(prompt: str, max_new_tokens: int | None = None) -> str:
    tok, model = _load()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens or settings.max_new_tokens,
        do_sample=False,
    )
    return tok.decode(output_ids[0], skip_special_tokens=True).strip()
