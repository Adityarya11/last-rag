from lastrag.data import _parse_ids, html_to_markdown


def md(html: str) -> list[str]:
    return html_to_markdown(html, "T").splitlines()[1:]  # drop "# T"


def test_headings_and_paragraphs_are_separated():
    html = (
        '<div data-component-type="text"><div>Intro text.</div></div>'
        '<div data-component-type="heading"><h2 id="x" level="3">Section</h2></div>'
        '<div data-component-type="heading"><h2 id="" level="4">To do it:</h2></div>'
        '<div data-component-type="text"><div>Body.</div></div>'
    )
    assert md(html) == ["Intro text.", "", "## Section", "", "### To do it:", "Body."]


def test_numbering_continues_across_screenshots_but_not_across_text():
    html = (
        "<ol><li>One</li><li>Two</li></ol>"
        '<div data-component-type="image"><img src="a.png" /></div>'
        "<ol><li>Three</li></ol>"
        "<div>Some text.</div>"
        "<ol><li>Fresh</li></ol>"
    )
    assert md(html) == ["1. One", "2. Two", "3. Three", "Some text.", "1. Fresh"]


def test_tabs_get_their_own_headings():
    html = (
        '<div class="tabs-wrapper"><div class="headings-wrapper">'
        '<div class="tab-heading heading-0">Wix Editor</div>'
        '<div class="tab-heading heading-1">Studio Editor</div></div>'
        '<div class="tab-contents-wrapper">'
        '<div class="tab-content"><ol><li>A</li></ol></div>'
        '<div class="tab-content"><ol><li>B</li></ol></div></div></div>'
    )
    assert md(html) == [
        "",
        "#### In Wix Editor:",
        "1. A",
        "",
        "#### In Studio Editor:",
        "1. B",
    ]


def test_table_rows_and_inline_markup():
    html = (
        "<table><tr><td><div><strong>Fees</strong></div></td>"
        "<td><div>2.6% +<br />0.20 USD</div></td></tr></table>"
        "<div>Click the <strong>Pages </strong>tab.</div>"
    )
    assert md(html) == ["| Fees | 2.6% + 0.20 USD |", "Click the Pages tab."]


def test_parse_ids_accepts_list_or_string():
    assert _parse_ids(["a", "b"]) == ["a", "b"]
    assert _parse_ids("['a', 'b']") == ["a", "b"]
