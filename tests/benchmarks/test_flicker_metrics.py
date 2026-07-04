"""Unit tests for the three pure flicker/commit-latency metrics (spec
§10.4), plus the grid parser and the M-8 delta-duplication arithmetic.
Everything here is hand-built sequences -- no server needed."""

from __future__ import annotations

import pytest
import yaml
from benchmarks.run_stabilizer_study import (
    STABILIZER_FIELD_TYPES,
    _hyp_text_at_tick,
    commit_latency_ms,
    commit_latency_word_latencies,
    common_prefix_len,
    delta_duplication_ratio,
    extra_chars,
    flicker_rate,
    make_temp_config,
    parse_grid,
    retracted_chars,
)

# -- common_prefix_len --


def test_common_prefix_len_worked_example():
    # "THE CAT" vs "THE CAR" share "THE CA" (6 chars) before diverging at
    # index 6 ('T' vs 'R').
    assert common_prefix_len("THE CAT", "THE CAR") == 6


def test_common_prefix_len_no_overlap():
    assert common_prefix_len("abc", "xyz") == 0


def test_common_prefix_len_identical():
    assert common_prefix_len("same", "same") == 4


# -- retracted_chars / flicker_rate --


def test_retracted_chars_worked_example_from_brief():
    # partials ["THE", "THE CAT", "THE CAR"]:
    #   ("THE", "THE CAT"): "THE CAT".startswith("THE") -> extends, +0
    #   ("THE CAT", "THE CAR"): does NOT extend; common_prefix_len == 6;
    #     len("THE CAT") == 7; retracted = 7 - 6 = 1
    # total: 1
    assert retracted_chars(["THE", "THE CAT", "THE CAR"]) == 1


def test_retracted_chars_pure_growth_is_zero():
    assert retracted_chars(["a", "a b", "a b c", "a b c d"]) == 0


def test_retracted_chars_no_partials_at_all():
    assert retracted_chars([]) == 0


def test_retracted_chars_single_partial_has_no_pairs():
    assert retracted_chars(["only one"]) == 0


def test_retracted_chars_volatile_shrink_then_regrow():
    # "hello world" -> "hello" (shrink: loses " world", 6 chars retracted)
    # -> "hello there" (from "hello": does "hello there" start with
    #    "hello"? yes -> extends, +0)
    partials = ["hello world", "hello", "hello there"]
    # pair 1: "hello".startswith("hello world")? no. common_prefix_len == 5.
    #   len("hello world") == 11. retracted = 11 - 5 = 6.
    # pair 2: "hello there".startswith("hello")? yes -> +0.
    assert retracted_chars(partials) == 6


def test_flicker_rate_worked_example():
    partials = ["THE", "THE CAT", "THE CAR"]
    final = "THE CAR"
    # retracted_chars == 1, len(final) == 7 -> 1/7
    assert flicker_rate(partials, final) == pytest.approx(1 / 7)


def test_flicker_rate_no_retraction_is_zero():
    assert flicker_rate(["a", "a b"], "a b") == 0.0


def test_flicker_rate_empty_final_does_not_divide_by_zero():
    # max(1, len("")) == 1 floor guard.
    assert flicker_rate([], "") == 0.0


# -- flicker over COMBINED (stable, volatile) tick sequences --
# Regression tests for the runner wiring bug where flicker_rate was fed
# volatile_text alone: volatile shrinks on every normal commit (words
# migrate to stable_text), making commits look like huge retractions.
# _score_native must feed the COMBINED user-visible hypothesis per tick.


def _combined_seq(ticks: list[tuple[str, str]]) -> list[str]:
    """The per-tick sequence _score_native feeds flicker_rate: the combined
    stable+volatile hypothesis, via the same helper it uses."""
    return [_hyp_text_at_tick(stable, volatile) for stable, volatile in ticks]


def test_flicker_zero_for_realistic_commit_progression_with_no_correction():
    # A realistic (stable, volatile) progression where words steadily
    # migrate volatile -> stable and NOTHING is ever corrected. The
    # volatile column alone shrinks at every commit ("the quick" -> "quick
    # brown" -> "brown fox" -> ""), which the OLD buggy wiring scored as
    # massive retraction; the combined hypothesis only ever grows.
    ticks = [
        ("", "the"),
        ("", "the quick"),
        ("the", "quick brown"),
        ("the quick", "brown fox"),
        ("the quick brown fox", ""),
    ]
    seq = _combined_seq(ticks)
    assert seq == [
        "the",
        "the quick",
        "the quick brown",
        "the quick brown fox",
        "the quick brown fox",
    ]
    assert flicker_rate(seq, "the quick brown fox") == 0.0
    # Sanity-check the bug this guards against: the volatile-only sequence
    # DOES report spurious retraction (this is what the runner must never
    # feed the metric).
    assert retracted_chars([v for _, v in ticks]) > 0


def test_flicker_positive_for_genuine_retraction_in_combined_hypothesis():
    # A GENUINE correction: the viewer saw "the cat", then the hypothesis
    # was revised to "the car".
    ticks = [
        ("", "the cat"),
        ("the", "car"),
        ("the car", ""),
    ]
    seq = _combined_seq(ticks)
    assert seq == ["the cat", "the car", "the car"]
    # pair ("the cat", "the car"): no extension; common prefix "the ca"
    # (6 chars); len("the cat") == 7 -> 1 retracted char.
    # pair ("the car", "the car"): identical -> extends -> 0.
    assert retracted_chars(seq) == 1
    assert flicker_rate(seq, "the car") == pytest.approx(1 / 7)


# -- commit_latency_ms --


def test_commit_latency_worked_example_single_partial_identical_to_final():
    # Single partial, identical to final, arriving at t=100: every word is
    # simultaneously "first seen" and "stable" at t=100 -> latency 0 each.
    partials_with_time = [(100.0, "hello world", "")]
    assert commit_latency_ms(partials_with_time, "hello world") == 0.0


def test_commit_latency_no_partials_at_all():
    # Nothing observed -> nothing to average -> 0.0 sentinel (documented).
    assert commit_latency_ms([], "hello world") == 0.0


def test_commit_latency_word_appears_volatile_then_becomes_stable():
    # "hello" first appears as volatile at t=0, becomes stable at t=200.
    # "world" first appears (volatile) at t=200, becomes stable at t=400.
    partials_with_time = [
        (0.0, "", "hello"),
        (200.0, "hello", "world"),
        (400.0, "hello world", ""),
    ]
    # hello: first_appear=0 (volatile at t=0), stable_time=200 -> latency 200
    # world: first_appear=200 (volatile at t=200), stable_time=400 -> latency 200
    assert commit_latency_ms(partials_with_time, "hello world") == pytest.approx(200.0)


def test_commit_latency_word_never_appears_before_final_is_excluded():
    # final has a 3rd word ("today") that never shows up in any partial's
    # stable or volatile text -- excluded, not counted as 0 or penalized.
    partials_with_time = [
        (0.0, "", "hello"),
        (100.0, "hello world", ""),
    ]
    # Only "hello" (index 0) ever reaches stable text (at t=100, first
    # appeared at t=0 -> latency 100). "world" (index 1) never appears in
    # ANY partial's stable_text at position 1 (stable_text at t=100 is
    # "hello world", so position 1 IS "world" -- let's make it truly
    # absent instead):
    partials_with_time = [
        (0.0, "", "hello"),
        (100.0, "hello", ""),
    ]
    # final = "hello world today": index 0 ("hello") stable at t=100,
    # first appeared t=0 -> latency 100. index 1 ("world") and index 2
    # ("today") never appear in any stable_text -> excluded.
    result = commit_latency_ms(partials_with_time, "hello world today")
    assert result == pytest.approx(100.0)


def test_commit_latency_volatile_shrink_then_regrow_still_tracks_stable_word():
    # volatile grows, shrinks (drops "world"), then the word that matters
    # ("hello") is committed to stable regardless.
    partials_with_time = [
        (0.0, "", "hello world"),
        (50.0, "", "hello"),  # volatile shrank; "hello" still first-seen at t=0
        (150.0, "hello", ""),  # now stable
    ]
    # "hello": first_appear = 0.0 (earliest volatile-hyp tick containing it
    # at position 0), stable_time = 150.0 -> latency 150.0
    assert commit_latency_ms(partials_with_time, "hello") == pytest.approx(150.0)


def test_commit_latency_casefold_insensitive_matching():
    partials_with_time = [(10.0, "Hello", "")]
    assert commit_latency_ms(partials_with_time, "hello") == 0.0


def test_commit_latency_empty_final_is_zero():
    assert commit_latency_ms([(0.0, "", "")], "") == 0.0


def test_commit_latency_word_latencies_exposes_included_word_count():
    # Same scenario as the excluded-word test above: only "hello" (index 0)
    # is ever observed becoming stable; "world"/"today" are excluded. The
    # list form exposes exactly the included words' latencies, so callers
    # can report len(...) as the mean's true denominator.
    partials_with_time = [
        (0.0, "", "hello"),
        (100.0, "hello", ""),
    ]
    latencies = commit_latency_word_latencies(partials_with_time, "hello world today")
    assert latencies == [pytest.approx(100.0)]


def test_commit_latency_word_latencies_empty_cases():
    assert commit_latency_word_latencies([], "hello") == []
    assert commit_latency_word_latencies([(0.0, "x", "")], "") == []


# -- M-8: delta-duplication arithmetic --


def test_extra_chars_wire_perfect_is_zero():
    assert extra_chars("The quick brown fox.", "The quick brown fox.") == 0


def test_extra_chars_shrinking_final_full_resend():
    # already-sent "the quick" (9 chars) plus a full-resend catch-up delta
    # of the whole final text -> joined duplicates the first 9 chars.
    joined = "the quick" + "The quick brown fox."
    completed = "The quick brown fox."
    assert extra_chars(joined, completed) == len(joined) - len(completed) == 9


def test_extra_chars_floors_at_zero_for_shorter_joined():
    assert extra_chars("abc", "abcdef") == 0


def test_delta_duplication_ratio_wire_perfect():
    assert delta_duplication_ratio("hello world", "hello world") == 0.0


def test_delta_duplication_ratio_worked_example():
    joined = "the quick" + "The quick brown fox."
    completed = "The quick brown fox."
    # extra_chars == 9, len(completed) == 20 -> 9/20 == 0.45
    assert delta_duplication_ratio(joined, completed) == pytest.approx(9 / 20)


def test_delta_duplication_ratio_empty_completed_is_zero():
    assert delta_duplication_ratio("anything", "") == 0.0


# -- --grid parser --


def test_parse_grid_cross_product():
    grid = parse_grid("min_partials=1,2,3;min_stable_ms=0,240,480")
    assert len(grid) == 9
    assert {"min_partials": 1, "min_stable_ms": 0.0} in grid
    assert {"min_partials": 3, "min_stable_ms": 480.0} in grid
    # types cast per STABILIZER_FIELD_TYPES
    for point in grid:
        assert isinstance(point["min_partials"], STABILIZER_FIELD_TYPES["min_partials"])
        assert isinstance(point["min_stable_ms"], STABILIZER_FIELD_TYPES["min_stable_ms"])


def test_parse_grid_single_axis():
    grid = parse_grid("min_partials=1,2")
    assert grid == [{"min_partials": 1}, {"min_partials": 2}]


def test_parse_grid_unknown_field_raises():
    with pytest.raises(ValueError, match="unknown stabilizer field"):
        parse_grid("bogus_field=1,2")


def test_parse_grid_malformed_clause_raises():
    with pytest.raises(ValueError, match="malformed"):
        parse_grid("min_partials")


def test_parse_grid_empty_string_raises():
    with pytest.raises(ValueError):
        parse_grid("")


# -- make_temp_config --


def test_make_temp_config_patches_only_stabilizer_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "server": {"host": "0.0.0.0", "port": 8000},
                "vad": {"kind": "energy", "threshold_dbfs": -40.0},
                "stabilizer": {"min_partials": 2, "min_stable_ms": 400.0},
                "backends": {"mock": {"type": "mock", "options": {"partial_interval_ms": 240}}},
                "models": {"mock": "mock"},
            }
        )
    )

    out_path = make_temp_config(str(base), {"min_partials": 3}, tmp_path)
    data = yaml.safe_load(out_path.read_text())

    # Only the overridden stabilizer field changed; the OTHER stabilizer
    # field survives (merged, not replaced).
    assert data["stabilizer"] == {"min_partials": 3, "min_stable_ms": 400.0}
    # Every non-stabilizer section is byte-for-byte untouched.
    assert data["server"] == {"host": "0.0.0.0", "port": 8000}
    assert data["vad"] == {"kind": "energy", "threshold_dbfs": -40.0}
    assert data["backends"] == {
        "mock": {"type": "mock", "options": {"partial_interval_ms": 240}}
    }
    assert data["models"] == {"mock": "mock"}
    # Written to a NEW file under tmpdir; the base config is not mutated.
    assert out_path != base
    assert out_path.parent == tmp_path
    assert yaml.safe_load(base.read_text())["stabilizer"]["min_partials"] == 2


def test_make_temp_config_adds_stabilizer_section_when_base_lacks_one(tmp_path):
    base = tmp_path / "no-stabilizer.yaml"
    base.write_text(yaml.safe_dump({"models": {"mock": "mock"}}))

    out_path = make_temp_config(str(base), {"min_stable_ms": 240.0}, tmp_path)
    data = yaml.safe_load(out_path.read_text())

    assert data["stabilizer"] == {"min_stable_ms": 240.0}
    assert data["models"] == {"mock": "mock"}
