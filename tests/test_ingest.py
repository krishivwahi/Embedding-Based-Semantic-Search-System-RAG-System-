import pytest

from src.ingest import chunk_text


def test_chunks_overlap():
    words = " ".join(str(i) for i in range(10))
    assert chunk_text(words, size=4, overlap=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9", "9"]


def test_short_text_is_one_chunk():
    assert chunk_text("a b c", size=500, overlap=50) == ["a b c"]


@pytest.mark.parametrize("overlap", [4, 5, -1])
def test_bad_overlap_raises(overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", size=4, overlap=overlap)
