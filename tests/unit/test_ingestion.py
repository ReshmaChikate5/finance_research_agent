"""Tests for the ingestion pipeline (chunker, load_chroma helpers, end-to-end into Chroma).

Run from the REPO ROOT:
    uv run --project agent_host --extra local python -m pytest tests/unit/test_ingestion.py

No network and no model download: filings are synthetic HTML shaped like real EDGAR
documents (hidden iXBRL header, table of contents that repeats every Item heading,
page numbers, financial tables, cross-reference lines), and embeddings use a fake embedder.
"""

import asyncio
import hashlib
import itertools
import math
import re

import pytest
from app.vectorstore.chroma_store import ChromaVectorStore

from ingestion.chunker import (
    ChunkStats,
    chunk_filing,
    chunk_sections,
    html_to_blocks,
    parse_sections,
    split_text,
)
from ingestion.load_chroma import Options, news_to_chunk, process_filing
from ingestion.sources import Filing, NewsItem

RISK_PARAS = [
    (
        "Our business depends on a limited number of suppliers. A shortage of components could delay "
        "shipments and reduce revenue. We have experienced supply constraints in the past and may "
        "again in the future, which could harm our results of operations and customer relationships."
    ),
    (
        "We face intense competition. Competitors may introduce products with better performance or "
        "lower prices. If we fail to keep pace with technology changes our market share could decline."
    ),
    (
        "Changes in export regulations could restrict our ability to sell certain products in some "
        "countries. See Item 7 for discussion of the effect on results. Compliance costs may increase."
    ),
]


def _p(text: str) -> str:
    return f'<div><span style="font-size:10pt">{text}</span></div>'


def make_10k() -> str:
    toc_rows = "".join(
        f"<tr><td>Item {i}.</td><td>{t}</td><td>{n}</td></tr>"
        for i, t, n in [
            ("1", "Business", 4),
            ("1A", "Risk Factors", 12),
            ("1B", "Unresolved Staff Comments", 20),
            ("2", "Properties", 21),
            ("7", "Management's Discussion and Analysis", 30),
            ("8", "Financial Statements and Supplementary Data", 45),
        ]
    )
    return f"""<html><head><title>x</title><style>p{{}}</style></head><body>
    <div style="display:none"><ix:header><ix:hidden>HIDDEN-XBRL-METADATA</ix:hidden></ix:header></div>
    <div>ACME CORP ANNUAL REPORT FORM 10-K</div>
    <div>Table of Contents</div>
    <table>{toc_rows}</table>
    <div>PART I</div>
    <div><span>Item 1.</span><span> Business</span></div>
    {_p("ACME designs and sells widgets to <b>enterprise</b> customers around the world.")}
    {
        _p(
            "We sell through <span>distributors</span> and directly. "
            "Our <ix:nonnumeric>fiscal year</ix:nonnumeric> ends in January."
        )
    }
    <div>12</div>
    <div>Item 1A. Risk Factors</div>
    {"".join(_p(t) for t in RISK_PARAS)}
    <div>Item 1B. Unresolved Staff Comments</div>
    {_p("None.")}
    <div>Item 2. Properties</div>
    {_p("Our headquarters is located in Springfield and we lease additional space elsewhere.")}
    <div>PART II</div>
    <div>Item 7. Management's Discussion and Analysis of Financial Condition</div>
    {_p("Revenue increased 12% compared with the prior year, driven by strong demand.")}
    <div>Item 1A. Risk Factors</div>
    {
        _p(
            "Item 1 of this report describes our business in detail and should be read with this section."
        )
    }
    <table>
      <tr><td>Fiscal year</td><td>2025</td><td>2024</td></tr>
      <tr><td>Revenue</td><td>$</td><td>1,234</td><td>$</td><td>1,100</td></tr>
      <tr><td>Net income</td><td>$</td><td>(56</td><td>)</td><td>40</td></tr>
    </table>
    <div>Item 8. Financial Statements and Supplementary Data</div>
    {_p("The financial statements are included below. Total assets were large.")}
    </body></html>"""


# ---------------------------------------------------------------------------
# html_to_blocks
# ---------------------------------------------------------------------------
def test_html_to_blocks_joins_inline_spans_and_drops_hidden_content():
    blocks = html_to_blocks(make_10k())
    assert "ACME designs and sells widgets to enterprise customers around the world." in blocks
    assert "Item 1. Business" in blocks  # two spans -> one line
    assert not any("HIDDEN-XBRL" in b for b in blocks)
    assert not any(b == "12" for b in blocks)  # page number dropped
    assert "Table of Contents" not in blocks


def test_table_rows_become_single_lines_with_currency_reattached():
    blocks = html_to_blocks(make_10k())
    assert "Revenue | 1,234 | 1,100" in blocks
    assert "Net income | (56) | 40" in blocks


def test_layout_table_cell_with_paragraphs_is_not_collapsed_into_one_line():
    html = "<html><body><table><tr><td><p>First paragraph.</p><p>Second paragraph.</p></td></tr></table></body></html>"
    assert html_to_blocks(html) == ["First paragraph.", "Second paragraph."]


# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------
def test_sections_found_in_order_and_toc_is_ignored():
    sections = parse_sections(make_10k(), "10-K")
    assert [s.key for s in sections] == ["1", "1A", "1B", "2", "7", "8"]
    business = sections[0]
    assert business.paragraphs[0].startswith("ACME designs and sells widgets")
    assert all(not re.fullmatch(r"\d+", p) for p in business.paragraphs)
    assert sections[1].title == "Risk Factors"
    assert not any(p.upper().startswith("PART ") for s in sections for p in s.paragraphs)
    assert sections[1].label == "Item 1A - Risk Factors"


def test_real_section_beats_toc_entry_and_body_crossref_does_not_split_it():
    risk = next(s for s in parse_sections(make_10k(), "10-K") if s.key == "1A")
    text = " ".join(risk.paragraphs)
    assert "limited number of suppliers" in text
    assert "export regulations" in text
    assert "Springfield" not in text


def test_item_cross_reference_sentences_are_not_headings():
    blocks = [
        "Item 1 of this report describes our business.",
        "Item 1A. Risk Factors",
        "text " * 50,
        "Item 2. Properties",
        "more " * 20,
    ]
    keys = [
        s.key
        for s in __import__("ingestion.chunker", fromlist=["x"]).sections_from_blocks(
            blocks, "10-K"
        )
    ]
    assert keys == ["1A", "2"]


def test_no_items_falls_back_to_full_document():
    sections = parse_sections(
        "<html><body><p>Just some text here.</p><p>And more text.</p></body></html>", "10-K"
    )
    assert [s.key for s in sections] == ["FULL"]
    assert sections[0].label == "Full document"


def test_10q_sections_are_part_aware():
    html = """<html><body>
      <div>PART I - FINANCIAL INFORMATION</div>
      <div>Item 1. Financial Statements</div><div>Statements text goes here and is long enough.</div>
      <div>Item 2. Management's Discussion and Analysis</div><div>MD&A text goes here.</div>
      <div>PART II - OTHER INFORMATION</div>
      <div>Item 1. Legal Proceedings</div><div>None at this time.</div>
      <div>Item 1A. Risk Factors</div><div>There were no material changes to risk factors.</div>
    </body></html>"""
    sections = parse_sections(html, "10-Q")
    assert [s.key for s in sections] == ["I-1", "I-2", "II-1", "II-1A"]
    assert sections[2].title == "Legal Proceedings"  # not confused with Part I Item 1
    assert sections[3].label == "Part II, Item 1A - Risk Factors"


# ---------------------------------------------------------------------------
# split_text
# ---------------------------------------------------------------------------
def test_chunks_respect_max_chars_and_keep_whole_paragraphs():
    paras = [f"Paragraph number {i}. " + "word " * 30 for i in range(12)]
    chunks = split_text(paras, max_chars=500, overlap=100)
    assert all(len(c) <= 500 for c in chunks)
    joined = "\n".join(chunks)
    assert all(p.strip() in joined for p in paras)  # nothing lost


def test_overlap_repeats_trailing_paragraph_at_start_of_next_chunk():
    paras = [f"Distinct paragraph {i} " + "x" * 80 for i in range(8)]  # 101 chars each
    chunks = split_text(paras, max_chars=450, overlap=120)  # holds 4 paragraphs, carries 1
    assert len(chunks) > 1
    for a, b in itertools.pairwise(chunks):
        assert b.splitlines()[0] == a.splitlines()[-1]


def test_oversized_paragraph_is_split_on_sentences_and_nothing_exceeds_limit():
    para = " ".join(f"This is sentence number {i} of a very long paragraph." for i in range(60))
    chunks = split_text([para], max_chars=300, overlap=0)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].startswith("This is sentence number 0")
    assert "\n" not in chunks[0]  # sentences of one paragraph are rejoined with spaces


def test_single_giant_token_stream_is_hard_split():
    chunks = split_text(["word " * 500], max_chars=200, overlap=0)
    assert all(len(c) <= 200 for c in chunks)


def test_empty_input_gives_no_chunks():
    assert split_text([], 500, 50) == []


# ---------------------------------------------------------------------------
# chunk_sections / chunk_filing
# ---------------------------------------------------------------------------
KW = {
    "ticker": "ACME",
    "form_type": "10-K",
    "accession_number": "0000000001-25-000001",
    "filing_date": "2025-02-26",
    "report_date": "2025-01-26",
    "company_name": "ACME CORP",
    "source_url": "https://example.invalid/acme.htm",
}


def test_chunks_carry_citation_metadata_header_and_stable_ids():
    chunks = chunk_filing(make_10k(), max_chars=300, overlap=50, **KW)
    first = chunks[0]
    assert first.id == "0000000001-25-000001:1:0000"
    assert first.text.startswith("[ACME 10-K | filed 2025-02-26 | Item 1 - Business]\n")
    assert first.metadata["section"] == "1" and first.metadata["section_title"] == "Business"
    assert first.metadata["doc_type"] == "filing"
    assert first.metadata["accession_number"] == "0000000001-25-000001"
    assert first.metadata["source_url"] == "https://example.invalid/acme.htm"
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.id for c in chunks] == [
        c.id for c in chunk_filing(make_10k(), max_chars=300, overlap=50, **KW)
    ]


def test_chunks_never_cross_section_boundaries():
    for c in chunk_filing(make_10k(), max_chars=300, overlap=50, **KW):
        body = c.text.split("\n", 1)[1]
        if c.metadata["section"] == "1":
            assert "export regulations" not in body and "Springfield" not in body


def test_skip_sections_and_empty_sections_are_omitted():
    chunks = chunk_filing(make_10k(), skip_sections=["1A"], **KW)
    assert "1A" not in {c.metadata["section"] for c in chunks}
    assert "1B" in {c.metadata["section"] for c in chunks}  # a lone "None." is kept: it is a fact


def test_token_fit_pass_resplits_chunks_that_exceed_the_model_window():
    sections = parse_sections(make_10k(), "10-K")
    stats = ChunkStats()

    def counter(texts):  # pretend 1 token per 2 characters (dense financial text)
        return [len(t) // 2 for t in texts]

    chunks = chunk_sections(
        sections,
        max_chars=800,
        overlap=60,
        token_counter=counter,
        max_tokens=100,
        stats=stats,
        **{k: v for k, v in KW.items()},
    )
    assert stats.resplit > 0
    assert all(counter([c.text])[0] <= 100 for c in chunks)
    plain = chunk_sections(sections, max_chars=800, overlap=60, **KW)
    assert len(chunks) > len(plain)


# ---------------------------------------------------------------------------
# End to end: filing -> chunks -> Chroma -> filtered query
# ---------------------------------------------------------------------------
class FakeEmbedder:
    name = "fake-hash-64"

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * 64
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


FILING = Filing(
    "ACME", "ACME CORP", 1, "10-K", "0000000001-25-000001", "2025-02-26", "2025-01-26", "acme.htm"
)


def test_end_to_end_into_chroma_with_section_filter(tmp_path):
    sections, chunks, _stats = process_filing(
        FILING, make_10k(), Options(max_chars=300, overlap=50)
    )
    assert [s.key for s in sections][:2] == ["1", "1A"]
    store = ChromaVectorStore(tmp_path / "db", "test", embedder=FakeEmbedder())
    assert asyncio.run(store.upsert(chunks)) == len(chunks)

    hits = asyncio.run(store.query("supplier shortage of components delay shipments", k=2))
    assert hits[0].metadata["section"] == "1A"
    only_properties = asyncio.run(store.query("headquarters", k=5, filters={"section": "2"}))
    assert {h.metadata["section"] for h in only_properties} == {"2"}

    # Re-running replaces instead of duplicating
    asyncio.run(store.upsert(chunks))
    assert asyncio.run(store.count()) == len(chunks)


def test_news_chunk_shape():
    item = NewsItem(
        "123",
        "NVDA",
        "Chipmaker beats estimates",
        "Revenue rose sharply.",
        "https://example.invalid/n",
        "Reuters",
        "2026-10-20",
    )
    chunk = news_to_chunk(item)
    assert chunk.id == "news:NVDA:123"
    assert chunk.text.startswith("[NVDA news | 2026-10-20 | Reuters]\nChipmaker beats estimates")
    assert chunk.metadata["doc_type"] == "news" and chunk.metadata["ticker"] == "NVDA"


def test_cli_dry_run_help_parses():
    from ingestion.load_chroma import parse_args

    args = parse_args(
        ["--tickers", "nvda", "amd", "--forms", "10-K", "--dry-run", "--skip-sections", "15", "16"]
    )
    assert args.tickers == ["nvda", "amd"] and args.dry_run and args.skip_sections == ["15", "16"]


# ---------------------------------------------------------------------------
# The real CLI flow, with fake SEC access and a fake model
# ---------------------------------------------------------------------------
class FakeIngestEmbedder(FakeEmbedder):
    max_tokens = 256

    def count_tokens(self, texts):
        return [len(t) // 3 for t in texts]  # dense text: 3 chars per token


class FakeEdgar:
    def __init__(self, *_args, **_kwargs):
        pass

    def list_filings(self, ticker, form_type, limit):
        return [FILING] if ticker == "ACME" else []

    def download_html(self, filing, refresh=False):
        return make_10k()

    def close(self):
        pass


@pytest.fixture
def cli(monkeypatch, tmp_path):
    from app.config import Settings

    import ingestion.embedder as embedder_module
    import ingestion.load_chroma as lc

    settings = Settings(vector_store="chroma", chroma_path=tmp_path / "chroma", _env_file=None)
    monkeypatch.setattr(lc, "get_settings", lambda: settings)
    monkeypatch.setattr(lc, "EdgarClient", FakeEdgar)
    fake_ingest = type("S", (), {"sec_user_agent": "t t@t.t", "finnhub_api_key": ""})()
    monkeypatch.setattr(lc, "IngestSettings", lambda: fake_ingest)
    monkeypatch.setattr(embedder_module, "get_embedder", lambda name: FakeIngestEmbedder())
    return lc, settings


def test_cli_dry_run_reports_sections_and_stores_nothing(cli, capsys):
    lc, settings = cli
    assert lc.main(["--tickers", "ACME", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "ACME 10-K filed 2025-02-26" in out
    assert "Risk Factors" in out and "Dry run only" in out
    assert not settings.chroma_path.exists()


def test_cli_full_run_loads_chroma_and_runs_smoke_query(cli, capsys):
    lc, settings = cli
    assert lc.main(["--tickers", "ACME", "--max-chars", "300", "--overlap", "50"]) == 0
    out = capsys.readouterr().out
    assert "upserted" in out and "Smoke test" in out
    store = ChromaVectorStore(
        settings.chroma_path, settings.chroma_collection, embedder=FakeEmbedder()
    )
    assert asyncio.run(store.count()) > 5


def test_cli_reset_deletes_the_old_database_first(cli, capsys):
    lc, settings = cli
    settings.chroma_path.mkdir(parents=True)
    (settings.chroma_path / "old_index.marker").write_text("stale")
    assert lc.main(["--tickers", "ACME", "--reset"]) == 0
    assert "Resetting" in capsys.readouterr().out
    assert not (settings.chroma_path / "old_index.marker").exists()
    store = ChromaVectorStore(
        settings.chroma_path, settings.chroma_collection, embedder=FakeEmbedder()
    )
    assert asyncio.run(store.count()) > 0


def test_cli_returns_nonzero_when_nothing_could_be_ingested(cli, capsys):
    lc, _ = cli
    assert lc.main(["--tickers", "NOPE"]) == 1
    assert "no 10-K filings found" in capsys.readouterr().err
