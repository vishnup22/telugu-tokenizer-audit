from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# model_id: HuggingFace repo. trust_remote_code required for models with custom architectures.
# use_sentencepiece: True when AutoTokenizer loads a broken stub — load tokenizer.model directly.
HF_MODELS: dict[str, dict] = {
    "hf-gpt2": {"model_id": "gpt2"},
    "sarvam-2b": {"model_id": "sarvamai/sarvam-1-v0.5"},
    "sarvam-105b": {"model_id": "sarvamai/sarvam-105b", "trust_remote_code": True},
    "telugu-gpt2": {"model_id": "pulipakav-1/dravidian-gpt2-telugu", "use_sentencepiece": True},
}


def _load_sentencepiece(model_id: str, token: str | None) -> object:
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(model_id, "tokenizer.model", token=token)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def get_tokenizers() -> tuple[dict, dict]:
    tokenizers: dict = {}
    versions: dict = {}

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        logger.warning("Skipping HF tokenizers: %s", exc)
        return tokenizers, versions

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    for name, cfg in HF_MODELS.items():
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
