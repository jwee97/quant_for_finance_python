"""A result's identity is the hash of exactly the configuration it depends on."""

from __future__ import annotations

from src.utils.config import CONFIG_FILES, CORE_NAMESPACES, GEN2_NAMESPACES, GEN3_NAMESPACES, load_config


def test_generation_one_and_two_identities_are_pinned():
    """Adding Generation 3 files must not rename a Generation 1 or 2 result (their reports cite these)."""
    config = load_config()
    assert config.fingerprint("core") == "12f9cfd14055"
    assert config.fingerprint("all") == "c25e66aa003a"


def test_generation_three_scope_covers_every_namespace_and_changes_with_its_files():
    config = load_config()
    assert set(CONFIG_FILES) == set(CORE_NAMESPACES) | set(GEN2_NAMESPACES) | set(GEN3_NAMESPACES)
    before = config.fingerprint("gen3")
    config.bayes = {**config.bayes, "probe": 1}
    assert config.fingerprint("gen3") != before
    assert config.fingerprint("all") == "c25e66aa003a"           # unaffected
