from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

HF_MODELS: dict[str, dict] = {
    "hf-gpt2": {"model_id": "gpt2"},
    "sarvam-2b": {"model_id": "sarvamai/sarvam-2b-v0.5"},
    "sarvam-105b": {"model_id": "sarvamai/sarvam-105b", "trust_remote_code": True},
    "telugu-gpt2": {"model_id": "pulipakav-1/dravidian-gpt2-telugu", "use_sentencepiece": True},
    "xlm-r": {"model_id": "xlm-roberta-base"},
    "indicbert": {"model_id": "ai4bharat/indic-bert"},
    "llama-3": {"model_id": "meta-llama/Meta-Llama-3-8B"},
}


def _load_sentencepiece(model_id: str, token: str | None) -> object:
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(model_id, "tokenizer.model", token=token)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def get_tokenizers(include: set[str] | None = None) -> tuple[dict, dict]:
    tokenizers: dict = {}
    versions: dict = {}

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        logger.warning("Skipping HF tokenizers: %s", exc)
        return tokenizers, versions

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    selected_names = include if include is not None else set(HF_MODELS)

    for name, cfg in HF_MODELS.items():
        if name not in selected_names:
            continue
        model_id = cfg["model_id"]
        try:
            if cfg.get("use_sentencepiece"):
                sp = _load_sentencepiece(model_id, token)

                def make_sp_counter(sp_model):
                    def count(text: str) -> int:
                        return len(sp_model.encode(text, out_type=str))
                    return count

                tokenizers[name] = make_sp_counter(sp)
                versions[name] = {
                    "library": f"sentencepiece=={__import__('sentencepiece').__version__}",
                    "model": model_id,
                    "vocab_size": sp.vocab_size(),
                }
            else:
                trust_remote_code = cfg.get("trust_remote_code", False)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    token=token,
                    trust_remote_code=trust_remote_code,
                )

                def make_counter(tok):
                    def count(text: str) -> int:
                        return len(tok.encode(text, add_special_tokens=False))
                    return count

                tokenizers[name] = make_counter(tokenizer)
                versions[name] = {
                    "library": f"transformers=={transformers.__version__}",
                    "model": model_id,
                    "revision": getattr(tokenizer, "_commit_hash", None) or "unknown",
                }
        except Exception as exc:
            logger.warning("Skipping %s: %s", name, exc)

    return tokenizers, versions
