"""Populate the local Chroma vector store from SEC filings (and optionally news).

Run from the REPO ROOT:

    # 1. Check section detection first: no model, no database, just fetch + chunk + report
    uv run --project agent_host --extra local python -m ingestion.load_chroma --tickers NVDA --dry-run

    # 2. Real load
    uv run --project agent_host --extra local python -m ingestion.load_chroma --tickers NVDA --forms 10-K --limit 1

    # More: several tickers/forms, news, or rebuild from scratch
    ... --tickers NVDA AMD --forms 10-K 10-Q --limit 2
    ... --tickers NVDA --news-days 7
    ... --reset            (delete the Chroma folder first; use after changing chunk settings)

Writes to the Chroma folder/collection from your .env (CHROMA_PATH, CHROMA_COLLECTION).
Chunk ids are deterministic, so re-running the same filing replaces its chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import statistics
import sys
from dataclasses import dataclass

from app.config import get_settings
from app.vectorstore.base import Chunk

from ingestion.chunker import (
    DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, ChunkStats, Section, chunk_sections, parse_sections,
)
from ingestion.sources import EdgarClient, Filing, IngestSettings, NewsItem, SourceError, fetch_news

logger = logging.getLogger("ingestion")

EXPECTED_10K = ("1", "1A", "7")  # a 10-K missing these means detection probably failed


@dataclass
class Options:
    max_chars: int = DEFAULT_MAX_CHARS
    overlap: int = DEFAULT_OVERLAP
    skip_sections: tuple[str, ...] = ()


def process_filing(
    filing: Filing, html: str, opts: Options, token_counter=None, max_tokens=None,
) -> tuple[list[Section], list[Chunk], ChunkStats]:
    """HTML -> sections -> chunks for one filing (pure function, easy to test)."""
    sections = parse_sections(html, filing.form_type)
    stats = ChunkStats()
    chunks = chunk_sections(
        sections,
        ticker=filing.ticker,
        form_type=filing.form_type,
        accession_number=filing.accession_number,
        filing_date=filing.filing_date,
        report_date=filing.report_date,
        company_name=filing.company_name,
        source_url=filing.url,
        max_chars=opts.max_chars,
        overlap=opts.overlap,
        skip_sections=opts.skip_sections,
        token_counter=token_counter,
        max_tokens=max_tokens,
        stats=stats,
    )
    return sections, chunks, stats


def news_to_chunk(item: NewsItem) -> Chunk:
    header = f"[{item.ticker} news | {item.published} | {item.source}]"
    body = "\n".join(p for p in (item.headline, item.summary[:1200]) if p)
    return Chunk(
        f"news:{item.ticker}:{item.id}",
        f"{header}\n{body}",
        {
            "doc_type": "news", "ticker": item.ticker, "published": item.published,
            "source": item.source, "headline": item.headline[:300], "source_url": item.url,
        },
    )


def _ascii(text: str) -> str:
    """Console-safe text (Windows consoles often can't print every Unicode character)."""
    return text.encode("ascii", "replace").decode("ascii")


def report_filing(filing: Filing, sections: list[Section], chunks: list[Chunk], stats: ChunkStats) -> None:
    per_section: dict[str, int] = {}
    for c in chunks:
        per_section[c.metadata["section"]] = per_section.get(c.metadata["section"], 0) + 1
    print(f"\n{filing.ticker} {filing.form_type} filed {filing.filing_date} ({filing.accession_number})")
    print(f"  {'section':<8} {'title':<55} {'chars':>9} {'chunks':>7}")
    for sec in sections:
        print(f"  {sec.key:<8} {_ascii(sec.title)[:55]:<55} {sec.chars:>9,} {per_section.get(sec.key, 0):>7}")
    if chunks:
        sizes = [len(c.text) for c in chunks]
        line = (f"  total: {len(sections)} sections, {len(chunks)} chunks, "
                f"avg {statistics.mean(sizes):.0f} chars, max {max(sizes)}")
        if stats.resplit:
            line += f", {stats.resplit} re-split to fit the model window"
        print(line)
    if sections and sections[0].key == "FULL":
        print("  WARNING: no Item headings detected; indexed as one full-document section "
              "(citations will not show an Item).")
    elif filing.form_type.upper().startswith("10-K"):
        missing = [k for k in EXPECTED_10K if k not in {s.key for s in sections}]
        if missing:
            print(f"  WARNING: expected sections not found: {', '.join(missing)}. "
                  "Section detection may have failed for this filing.")


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    ingest = IngestSettings()
    opts = Options(args.max_chars, args.overlap, tuple(args.skip_sections))
    tickers = [t.upper() for t in args.tickers]
    forms = [f.upper() for f in args.forms]

    store = embedder = counter = max_tokens = None
    if not args.dry_run:
        from app.vectorstore.chroma_store import ChromaVectorStore  # heavy imports only when needed
        from ingestion.embedder import get_embedder

        if settings.vector_store != "chroma":
            print(f"Note: VECTOR_STORE={settings.vector_store}, but this script always loads local Chroma.")
        if args.reset and settings.chroma_path.exists():
            print(f"Resetting: deleting {settings.chroma_path}")
            try:
                shutil.rmtree(settings.chroma_path)
            except OSError as exc:
                raise SourceError(
                    f"Could not delete {settings.chroma_path} ({type(exc).__name__}). Stop anything "
                    "using it (the agent host, another ingestion run) and try again."
                ) from exc
        print(f"Loading embedding model {settings.embedding_model} (first run downloads ~90 MB)...")
        embedder = get_embedder(settings.embedding_model)
        max_tokens = embedder.max_tokens
        counter = embedder.count_tokens
        store = ChromaVectorStore(
            settings.chroma_path, settings.chroma_collection, embedder=embedder,
        )
        print(f"Chroma: {settings.chroma_path} (collection '{settings.chroma_collection}'), "
              f"model window {max_tokens} tokens")

    edgar = EdgarClient(ingest.sec_user_agent)
    ingested = 0
    problems: list[str] = []
    try:
        for ticker in tickers:
            for form in forms:
                try:
                    filings = edgar.list_filings(ticker, form, args.limit)
                except SourceError as exc:
                    problems.append(f"{ticker} {form}: {exc}")
                    continue
                if not filings:
                    problems.append(f"{ticker}: no {form} filings found")
                    continue
                for filing in filings:
                    try:
                        html = edgar.download_html(filing, refresh=args.refresh)
                    except SourceError as exc:
                        problems.append(f"{ticker} {filing.accession_number}: {exc}")
                        continue
                    sections, chunks, stats = process_filing(filing, html, opts, counter, max_tokens)
                    report_filing(filing, sections, chunks, stats)
                    if store and chunks:
                        written = await store.upsert(chunks)
                        ingested += written
                        print(f"  -> upserted {written} chunks")

            if args.news_days and not args.dry_run:
                try:
                    items = fetch_news(ticker, args.news_days, ingest.finnhub_api_key)
                except SourceError as exc:
                    problems.append(f"{ticker} news: {exc}")
                    continue
                news_chunks = [news_to_chunk(i) for i in items]
                if news_chunks:
                    ingested += await store.upsert(news_chunks)
                print(f"\n{ticker} news: upserted {len(news_chunks)} items from the last {args.news_days} days")
    finally:
        edgar.close()

    if store:
        total = await store.count()
        print(f"\nDone. Upserted {ingested} chunks this run; collection now holds {total}.")
        if tickers and total:
            hits = await store.query(
                "What are the main risk factors?", k=3, filters={"ticker": tickers[0]},
            )
            print(f"\nSmoke test: 'What are the main risk factors?' filtered to {tickers[0]}")
            for h in hits:
                lines = h.text.splitlines()  # line 0 is the citation header
                snippet = _ascii(lines[1][:90]) if len(lines) > 1 else ""
                print(f"  {h.score:.3f}  [{h.metadata.get('section', '-')}] {snippet}")
    elif args.dry_run:
        print("\nDry run only: nothing embedded or stored.")

    for p in problems:
        print(f"PROBLEM: {_ascii(p)}", file=sys.stderr)
    return 1 if problems and not ingested and not args.dry_run else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load SEC filings (and news) into local Chroma.")
    p.add_argument("--tickers", nargs="+", default=["NVDA"], help="default: NVDA")
    p.add_argument("--forms", nargs="+", default=["10-K"], help="e.g. 10-K 10-Q (default: 10-K)")
    p.add_argument("--limit", type=int, default=1, help="filings per ticker per form, newest first (default 1)")
    p.add_argument("--news-days", type=int, default=0, help="also load Finnhub news from the last N days")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    p.add_argument("--skip-sections", nargs="*", default=[], help="section keys to skip, e.g. 15 16")
    p.add_argument("--dry-run", action="store_true", help="fetch + chunk + report only")
    p.add_argument("--reset", action="store_true", help="delete the Chroma folder before loading")
    p.add_argument("--refresh", action="store_true", help="re-download filings instead of using .cache/")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except SourceError as exc:
        print(f"ERROR: {_ascii(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
