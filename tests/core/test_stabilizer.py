from stt_server.config.settings import StabilizerConfig
from stt_server.core.stabilizer import Stabilizer

CFG = StabilizerConfig(min_partials=2, min_stable_ms=400.0)


def test_commit_requires_survival_and_time():
    st = Stabilizer(CFG)
    u1 = st.update("i", now_ms=0)
    assert (u1.stable_text, u1.volatile_text) == ("", "i")   # seen once, 0 ms
    u2 = st.update("i want", now_ms=300)
    assert u2.stable_text == ""                              # "i" x2 but only 300 ms
    u3 = st.update("i want a", now_ms=500)
    assert u3.stable_text == "i"                             # x3 and 500 ms
    assert u3.newly_committed == "i"
    assert u3.volatile_text == "want a"
    u4 = st.update("i want a coffee", now_ms=900)
    assert u4.stable_text == "i want a"                      # "want","a" now qualify
    assert u4.newly_committed == "want a"


def test_changed_token_resets_survival():
    st = Stabilizer(CFG)
    st.update("i went", now_ms=0)
    st.update("i want", now_ms=500)      # "went"->"want": survival restarts
    u = st.update("i want", now_ms=600)  # "want" x2 but only 100 ms old
    assert u.stable_text == "i"
    u = st.update("i want more", now_ms=1000)
    assert u.stable_text == "i want"


def test_committed_prefix_never_shrinks():
    st = Stabilizer(CFG)
    st.update("alpha beta", now_ms=0)
    st.update("alpha beta", now_ms=500)
    u = st.update("alpha beta", now_ms=1000)
    assert u.stable_text == "alpha beta"
    u = st.update("alfa", now_ms=1500)   # backend contradicts committed text
    assert u.stable_text == "alpha beta"
    assert u.volatile_text == ""
    assert u.newly_committed == ""


def test_monotonic_stable_prefix_property():
    """Committed text is always a prefix-extension of the previous committed text."""
    partials = ["the", "the cat", "the can", "the can of", "the cat sat", "the cat sat down"]
    st = Stabilizer(CFG)
    prev = ""
    for i, p in enumerate(partials):
        u = st.update(p, now_ms=i * 450.0)
        assert u.stable_text.startswith(prev)
        prev = u.stable_text


def test_reset_clears_state():
    st = Stabilizer(CFG)
    st.update("hello world", now_ms=0)
    st.update("hello world", now_ms=500)
    st.reset()
    u = st.update("goodbye", now_ms=1000)
    assert (u.stable_text, u.volatile_text) == ("", "goodbye")


def test_empty_update_is_harmless():
    st = Stabilizer(CFG)
    st.update("hello world", now_ms=0)
    u = st.update("", now_ms=500)
    assert u.stable_text == ""      # nothing committed yet, nothing invented
    assert u.volatile_text == ""
    assert u.newly_committed == ""
