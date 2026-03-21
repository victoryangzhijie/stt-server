from __future__ import annotations

import pytest


class TestHypothesisBuffer:
    """Tests for _HypothesisBuffer — word-level local agreement."""

    def _make_buffer(self):
        from backends.whisper import _HypothesisBuffer
        return _HypothesisBuffer()

    def test_flush_returns_common_prefix(self):
        """Two identical inserts → words confirmed on flush."""
        buf = self._make_buffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        buf.insert(words, offset=0.0)
        committed = buf.flush()
        assert committed == []
        buf.insert(words, offset=0.0)
        committed = buf.flush()
        assert len(committed) == 2
        assert committed[0][2] == "hello"
        assert committed[1][2] == "world"

    def test_unstable_text_not_committed(self):
        """Differing inserts → nothing committed."""
        buf = self._make_buffer()
        buf.insert([(0.0, 0.5, "hello")], offset=0.0)
        buf.flush()
        buf.insert([(0.0, 0.5, "goodbye")], offset=0.0)
        committed = buf.flush()
        assert committed == []

    def test_partial_agreement(self):
        """Only the matching prefix is committed."""
        buf = self._make_buffer()
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0.0)
        buf.flush()
        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "there")], offset=0.0)
        committed = buf.flush()
        assert len(committed) == 1
        assert committed[0][2] == "hello"

    def test_complete_returns_buffered(self):
        """complete() returns unconfirmed words in buffer."""
        buf = self._make_buffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        buf.insert(words, offset=0.0)
        buf.flush()
        buffered = buf.complete()
        assert len(buffered) == 2
        assert buffered[0][2] == "hello"

    def test_pop_committed_removes_old(self):
        """pop_commited trims committed words before a timestamp."""
        buf = self._make_buffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        buf.insert(words, offset=0.0)
        buf.flush()
        buf.insert(words, offset=0.0)
        buf.flush()
        buf.pop_commited(0.6)
        assert len(buf.commited_in_buffer) == 1
        assert buf.commited_in_buffer[0][2] == "world"

    def test_insert_filters_old_words(self):
        """insert() drops words before last_commited_time."""
        buf = self._make_buffer()
        words1 = [(0.0, 0.5, "hello")]
        buf.insert(words1, offset=0.0)
        buf.flush()
        buf.insert(words1, offset=0.0)
        buf.flush()
        words2 = [(0.0, 0.3, "old"), (0.5, 1.0, "world")]
        buf.insert(words2, offset=0.0)
        assert all(w[0] >= buf.last_commited_time - 0.1 for w in buf.new)
