import pytest

from lastrag.chunking import Chunk
from lastrag.index import index_fields, indexed_texts, table_name


def chunk(i: int, text: str = "Click Save.") -> Chunk:
    return Chunk(f"a:{i}", "a", "article", "Title", text)


def test_contextual_table_is_separate_from_the_baseline():
    plain = table_name("html", "section")
    ctx = table_name("html", "section", contextual=True)
    assert plain == "html_section"
    assert ctx == "html_section_ctx"


def test_every_flag_combination_gets_its_own_table():
    names = {
        table_name("html", "section", ctx, title)
        for ctx in (False, True)
        for title in (True, False)
    }
    assert len(names) == 4


def test_title_flag_controls_whether_the_title_is_indexed():
    embedded, searchable = index_fields([chunk(0)], ["body"], title=True)
    assert embedded == ["title: Title | text: body"]
    assert searchable == ["Title\nbody"]
    embedded, searchable = index_fields([chunk(0)], ["body"], title=False)
    assert embedded == ["title: none | text: body"]
    assert searchable == ["body"]


def test_without_contexts_chunks_are_indexed_as_they_are():
    assert indexed_texts([chunk(0)], None) == ["Click Save."]


def test_context_goes_in_front_of_the_chunk():
    out = indexed_texts([chunk(0)], {"a:0": "Part of the domain guide."})
    assert out == ["Part of the domain guide.\n\nClick Save."]


def test_missing_context_raises_instead_of_indexing_a_bare_chunk():
    with pytest.raises(KeyError):
        indexed_texts([chunk(0), chunk(1)], {"a:0": "only the first"})
