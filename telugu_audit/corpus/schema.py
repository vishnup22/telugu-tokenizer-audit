from __future__ import annotations

from dataclasses import dataclass

DEFAULT_REQUIRED_REGISTERS = ("native_formal", "native_informal", "romanized_informal")
DEFAULT_OPTIONAL_REGISTERS = ("english_wiki", "tenglish_informal")


@dataclass(frozen=True)
class LanguageProfile:
    name: str
    code: str
    prefix: str
    primary: bool
    native_formal_register: str
    native_informal_register: str
    romanized_informal_register: str
    tenglish_register: str | None = None
    minimal_pairs_path: str | None = None


VALID_MORPH_TYPES = frozenset(
    {
        "base_noun",
        "case_suffix",
        "case_suffix_with_sandhi",
        "plural_suffix",
        "honorific_suffix",
        "compound_with_sandhi",
        "borrowed_base",
        "borrowed_plus_case_suffix",
        "verb_agglutination_chain",
    }
)

HINDI_VALID_MORPH_TYPES = frozenset(
    {
        "base_noun",
        "oblique_case_form",
        "plural_form_direct",
        "plural_form_oblique",
        "honorific_form",
        "pronoun_oblique_form",
        "verb_participle_form",
        "verb_inflection_chain",
        "borrowed_form",
    }
)

MORPH_TYPES_BY_LANGUAGE: dict[str, frozenset[str]] = {
    "telugu": VALID_MORPH_TYPES,
    "hindi": HINDI_VALID_MORPH_TYPES,
}


def get_valid_morph_types(language: str | None) -> frozenset[str]:
    """Return the valid morph_type vocabulary for a language, defaulting to Telugu's."""
    if language is None:
        return VALID_MORPH_TYPES
    return MORPH_TYPES_BY_LANGUAGE.get(language, VALID_MORPH_TYPES)


def _prefixed(prefix: str, register: str) -> str:
    return f"{prefix}_{register}" if prefix else register


def get_language_profiles(config: dict | None = None) -> list[LanguageProfile]:
    config = config or {}
    raw_profiles = config.get("languages")
    if not raw_profiles:
        return [
            LanguageProfile(
                name="telugu",
                code="te",
                prefix="",
                primary=True,
                native_formal_register="native_formal",
                native_informal_register="native_informal",
                romanized_informal_register="romanized_informal",
                tenglish_register="tenglish_informal",
            )
        ]

    profiles: list[LanguageProfile] = []
    for idx, entry in enumerate(raw_profiles):
        name = entry["name"]
        code = entry.get("code", name)
        prefix = entry.get("prefix", code)
        primary = bool(entry.get("primary", idx == 0))
        include_tenglish = bool(entry.get("include_tenglish", False))
        tenglish_register = entry.get("tenglish_register")
        if include_tenglish and not tenglish_register:
            tenglish_register = _prefixed(prefix, "tenglish_informal")

        profiles.append(
            LanguageProfile(
                name=name,
                code=code,
                prefix=prefix,
                primary=primary,
                native_formal_register=entry.get(
                    "native_formal_register", _prefixed(prefix, "native_formal")
                ),
                native_informal_register=entry.get(
                    "native_informal_register", _prefixed(prefix, "native_informal")
                ),
                romanized_informal_register=entry.get(
                    "romanized_informal_register",
                    _prefixed(prefix, "romanized_informal"),
                ),
                tenglish_register=tenglish_register,
                minimal_pairs_path=entry.get("minimal_pairs_path"),
            )
        )

    if not any(profile.primary for profile in profiles):
        first = profiles[0]
        profiles[0] = LanguageProfile(
            name=first.name,
            code=first.code,
            prefix=first.prefix,
            primary=True,
            native_formal_register=first.native_formal_register,
            native_informal_register=first.native_informal_register,
            romanized_informal_register=first.romanized_informal_register,
            tenglish_register=first.tenglish_register,
            minimal_pairs_path=first.minimal_pairs_path,
        )
    return profiles


def get_primary_language_profile(config: dict | None = None) -> LanguageProfile:
    profiles = get_language_profiles(config)
    for profile in profiles:
        if profile.primary:
            return profile
    return profiles[0]


def get_required_registers(config: dict | None = None) -> tuple[str, ...]:
    profiles = get_language_profiles(config)
    return tuple(
        register
        for profile in profiles
        for register in (
            profile.native_formal_register,
            profile.native_informal_register,
            profile.romanized_informal_register,
        )
    )


def get_optional_registers(config: dict | None = None) -> tuple[str, ...]:
    config = config or {}
    optional: list[str] = []
    english_register = config.get("english_register", "english_wiki")
    if english_register:
        optional.append(english_register)

    for profile in get_language_profiles(config):
        if profile.tenglish_register:
            optional.append(profile.tenglish_register)

    seen: set[str] = set()
    deduped: list[str] = []
    for register in optional:
        if register not in seen:
            seen.add(register)
            deduped.append(register)
    return tuple(deduped)


def get_allowed_registers(config: dict | None = None) -> tuple[str, ...]:
    return get_required_registers(config) + get_optional_registers(config)
