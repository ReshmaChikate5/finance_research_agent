"""Section-aware chunking for SEC filings (10-K, 10-Q, and a generic fallback).

Pipeline:  HTML  ->  text blocks  ->  Items/sections  ->  paragraph-packed chunks.

Why not fixed-size splitting: filings have real semantic boundaries ("Item 1A. Risk
Factors"), and the agent must cite WHERE a claim came from. Every chunk therefore carries
its section in metadata and in a short header line, and chunks never cross section
boundaries. Inside a section, whole paragraphs (or table rows) are packed up to
`max_chars`, with a little overlap between neighbours.

Known limits (by design, kept simple):
  * Text before the first Item heading (cover page, forward-looking-statements preamble)
    is not indexed.
  * A table split across chunks loses its header row in the later chunks.
  * Item detection is heuristic. Use `python -m ingestion.load_chroma --dry-run` to see
    exactly which sections were found before embedding anything.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from app.vectorstore.base import Chunk, MetadataValue

DEFAULT_MAX_CHARS = 800  # ~200 tokens of prose; financial text tokenizes denser, see fit pass
DEFAULT_OVERLAP = 120
MIN_CHUNK_CHARS = 20

TokenCounter = Callable[[Sequence[str]], list[int]]

# ---------------------------------------------------------------------------
# 1. HTML -> text blocks
# ---------------------------------------------------------------------------
_BLOCK_TAGS = {
    "html", "body", "div", "p", "li", "ul", "ol", "table", "thead", "tbody", "tfoot",
    "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote", "pre", "center",
}
_WS = re.compile(r"[\s ​]+")
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)
_NOISE = re.compile(r"^(page\s*)?\d{1,3}$|^[A-Z]-\d{1,3}$|^[|\s]*$", re.I)
_NOISE_PHRASES = {"table of contents", "index"}


def _collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _row_text(row: Tag) -> str:
    """One table row -> 'cell | cell | cell' (currency/percent fragments are re-attached)."""
    parts: list[str] = []
    for cell in row.find_all(["td", "th"], recursive=False):
        text = _collapse(cell.get_text(" "))
        if not text or text == "$":
            continue
        if text in (")", "%", ")%") and parts:
            parts[-1] += text
        else:
            parts.append(text)
    return " | ".join(parts)


def _walk(node: Tag, out: list[str], buf: list[str]) -> None:
    def flush() -> None:
        text = _collapse("".join(buf))
        buf.clear()
        if text:
            out.append(text)

    for child in node.children:
        if type(child) is NavigableString:  # skips comments, doctype, CDATA
            buf.append(str(child))
        elif isinstance(child, Tag):
            name = child.name
            if name == "br":
                flush()
            elif name == "tr":
                flush()
                cells = child.find_all(["td", "th"], recursive=False)
                if len(cells) == 1 and cells[0].find(["p", "div", "table", "li"]):
                    _walk(cells[0], out, buf)  # layout table: treat the cell as a container
                    flush()
                else:
                    line = _row_text(child)
                    if line:
                        out.append(line)
            elif name in _BLOCK_TAGS:
                flush()
                _walk(child, out, buf)
                flush()
            else:  # inline (span, a, b, ix:nonnumeric, ...): keep accumulating
                _walk(child, out, buf)
    # caller flushes at block boundaries


def html_to_blocks(html: str) -> list[str]:
    """Convert filing HTML into a list of clean text lines (paragraphs / table rows)."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "noscript", "ix:header"]):
        if not tag.decomposed:
            tag.decompose()
    for tag in soup.find_all(style=_HIDDEN):
        if not tag.decomposed:
            tag.decompose()

    out: list[str] = []
    buf: list[str] = []
    root = soup.body or soup
    _walk(root, out, buf)
    tail = _collapse("".join(buf))
    if tail:
        out.append(tail)
    return [b for b in out if not _NOISE.match(b) and b.lower() not in _NOISE_PHRASES]


# ---------------------------------------------------------------------------
# 2. Blocks -> sections
# ---------------------------------------------------------------------------
TITLES_10K = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity and Related Stockholder Matters",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules", "16": "Form 10-K Summary",
}
TITLES_10Q = {
    "I-1": "Financial Statements", "I-2": "Management's Discussion and Analysis (MD&A)",
    "I-3": "Quantitative and Qualitative Disclosures About Market Risk",
    "I-4": "Controls and Procedures", "II-1": "Legal Proceedings", "II-1A": "Risk Factors",
    "II-2": "Unregistered Sales of Equity Securities and Use of Proceeds",
    "II-3": "Defaults Upon Senior Securities", "II-4": "Mine Safety Disclosures",
    "II-5": "Other Information", "II-6": "Exhibits",
}

_ITEM_RE = re.compile(r"^item\s*(\d{1,2}(?:\.\d{2})?[a-c]?)\b\s*(.*)$", re.I)
_PART_RE = re.compile(r"^part\s+(iv|iii|ii|i)\b(.*)$", re.I)


@dataclass(frozen=True)
class Section:
    key: str  # "1A" (10-K), "II-1A" (10-Q), "2.02" (8-K), or "FULL"
    title: str
    label: str  # human-readable, used in chunk headers
    paragraphs: tuple[str, ...]

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.paragraphs)


@dataclass(frozen=True)
class _Candidate:
    idx: int
    key: str
    rest: str
    part: str | None


def _parse_part(line: str) -> str | None:
    if len(line) > 80:
        return None
    m = _PART_RE.match(line)
    if not m:
        return None
    rest = m.group(2).strip()
    if rest and not (rest[0] in ".:-–—|" or rest[0].isupper()):
        return None  # "Part II, Item 7 of this report..." is a cross-reference, not a heading
    return m.group(1).upper()


def _parse_item(line: str) -> tuple[str, str] | None:
    if len(line) > 220:
        return None
    m = _ITEM_RE.match(line)
    if not m:
        return None
    rest = m.group(2).strip().lstrip(".:-–—| ").strip()
    if rest and rest[0].islower():
        return None  # "Item 1 of this report describes..." is body text
    if len(rest.split()) > 25:
        return None
    return m.group(1).upper(), rest


_PART_RANK = {"I": 1, "II": 2, "III": 3, "IV": 4}
_KEY_RE = re.compile(r"^(\d+(?:\.\d+)?)([A-C]?)$")


def _order(key: str) -> tuple[int, float, str]:
    """Canonical sort position of a section key: '1A' < '1B' < '2' < '7A'; 'I-2' < 'II-1'."""
    part = 0
    if "-" in key:
        prefix, key = key.split("-", 1)
        part = _PART_RANK.get(prefix, 0)
    m = _KEY_RE.match(key)
    return (part, float(m.group(1)), m.group(2)) if m else (part, 0.0, key)


def _titles_for(form_type: str) -> dict[str, str]:
    form = form_type.upper()
    if form.startswith("10-Q"):
        return TITLES_10Q
    if form.startswith("10-K"):
        return TITLES_10K
    return {}


def parse_sections(html: str, form_type: str = "10-K") -> list[Section]:
    """Split a filing into its Item sections (or one FULL section if none are detected)."""
    return sections_from_blocks(html_to_blocks(html), form_type)


def sections_from_blocks(blocks: Sequence[str], form_type: str = "10-K") -> list[Section]:
    use_parts = form_type.upper().startswith("10-Q")
    titles = _titles_for(form_type)

    # Find every line that looks like an Item heading (the table of contents has them too).
    candidates: list[_Candidate] = []
    part: str | None = None
    for idx, line in enumerate(blocks):
        new_part = _parse_part(line)
        if new_part:
            part = new_part
            continue
        parsed = _parse_item(line)
        if parsed:
            item, rest = parsed
            key = f"{part}-{item}" if use_parts and part else item
            candidates.append(_Candidate(idx, key, rest, part))

    # Cumulative character offsets, to measure how much text follows each candidate.
    offsets = [0]
    for b in blocks:
        offsets.append(offsets[-1] + len(b) + 1)

    # Every Item appears twice: in the table of contents and in the body. Real filings also
    # have legitimately tiny sections ("Item 1B ... None."), so "the candidate followed by the
    # most text" is NOT enough: a TOC row can out-weigh a 5-character body section.
    # Instead choose the chain of headings that (a) appears in canonical Item order and
    # (b) covers the most text. The TOC cannot win: its entries come before the body, so
    # taking a TOC entry forces dropping every earlier-numbered body section.
    n = len(candidates)
    spans = [
        1 + offsets[(candidates[i + 1].idx if i + 1 < n else len(blocks))] - offsets[candidates[i].idx]
        for i in range(n)
    ]
    orders = [_order(c.key) for c in candidates]
    score = list(spans)
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if orders[j] < orders[i] and score[j] + spans[i] > score[i]:
                score[i], prev[i] = score[j] + spans[i], j
    selected_idx: list[int] = []
    k = max(range(n), key=score.__getitem__) if n else -1
    while k != -1:
        selected_idx.append(k)
        k = prev[k]
    selected = [candidates[i] for i in reversed(selected_idx)]
    if len(selected) < 2:
        return [Section("FULL", "Full document", "Full document", tuple(blocks))] if blocks else []

    sections: list[Section] = []
    for i, cand in enumerate(selected):
        end_idx = selected[i + 1].idx if i + 1 < len(selected) else len(blocks)
        body = tuple(b for b in blocks[cand.idx + 1 : end_idx] if _parse_part(b) is None)
        item = cand.key.split("-", 1)[1] if use_parts and "-" in cand.key else cand.key
        title = titles.get(cand.key) or cand.rest.split("|")[0].strip() or "Untitled"
        if use_parts and cand.part:
            label = f"Part {cand.part}, Item {item} - {title}"
        else:
            label = f"Item {item} - {title}"
        sections.append(Section(cand.key, title, label, body))
    return sections


# ---------------------------------------------------------------------------
# 3. Paragraph packing
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“(A-Z0-9])")


def _hard_split(text: str, limit: int) -> Iterator[str]:
    current: list[str] = []
    size = 0
    for word in text.split():
        if current and size + len(word) + 1 > limit:
            yield " ".join(current)
            current, size = [], 0
        current.append(word)
        size += len(word) + 1
    if current:
        yield " ".join(current)


def _units(paragraphs: Sequence[str], max_chars: int) -> Iterator[tuple[str, bool]]:
    """Yield (text, starts_new_paragraph). Oversized paragraphs are cut at sentence ends."""
    for para in paragraphs:
        if len(para) <= max_chars:
            yield para, True
            continue
        first = True
        for sentence in _SENT_SPLIT.split(para):
            pieces = [sentence] if len(sentence) <= max_chars else list(_hard_split(sentence, max_chars))
            for piece in pieces:
                yield piece, first
                first = False


def _render(items: Sequence[tuple[str, bool]]) -> str:
    text = ""
    for i, (piece, new_para) in enumerate(items):
        text += piece if i == 0 else ("\n" if new_para else " ") + piece
    return text


def _length(items: Sequence[tuple[str, bool]]) -> int:
    return sum(len(t) for t, _ in items) + max(len(items) - 1, 0)


def split_text(
    paragraphs: Sequence[str],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Pack paragraphs into chunks of at most ~max_chars, repeating up to `overlap` chars
    of trailing paragraphs at the start of the next chunk."""
    overlap = max(0, min(overlap, max_chars // 3))
    chunks: list[str] = []
    cur: list[tuple[str, bool]] = []
    for unit in _units(paragraphs, max_chars):
        if cur and _length(cur) + 1 + len(unit[0]) > max_chars:
            chunks.append(_render(cur))
            carry: list[tuple[str, bool]] = []
            for item in reversed(cur):
                if _length([item, *carry]) > overlap:
                    break
                carry.insert(0, item)
            cur = carry if _length(carry) + 1 + len(unit[0]) <= max_chars else []
        cur.append(unit)
    if cur:
        chunks.append(_render(cur))
    return chunks


# ---------------------------------------------------------------------------
# 4. Sections -> Chunks (with a token-limit safety pass)
# ---------------------------------------------------------------------------
@dataclass
class ChunkStats:
    resplit: int = 0  # chunks that exceeded the model's token window and were split again


def _refit(
    body: str, tokens: int, header: str, overlap: int, counter: TokenCounter,
    max_tokens: int, stats: ChunkStats | None, depth: int = 0,
) -> list[str]:
    """Re-split a chunk whose header+body exceeds the embedding model's token window."""
    if tokens <= max_tokens or depth >= 4 or len(body) < 80:
        return [body]
    new_max = max(60, int(len(body) * (max_tokens / tokens) * 0.85))
    parts = split_text(body.split("\n"), new_max, min(overlap, new_max // 5))
    if len(parts) < 2:
        return [body]
    if stats is not None and depth == 0:
        stats.resplit += 1
    counts = counter([f"{header}\n{p}" for p in parts])
    out: list[str] = []
    for part, n in zip(parts, counts):
        out.extend(_refit(part, n, header, overlap, counter, max_tokens, stats, depth + 1))
    return out


def _bodies(
    paragraphs: Sequence[str], header: str, max_chars: int, overlap: int,
    counter: TokenCounter | None, max_tokens: int | None, stats: ChunkStats | None,
) -> list[str]:
    bodies = split_text(paragraphs, max_chars, overlap)
    if counter is None or max_tokens is None or not bodies:
        return bodies
    counts = counter([f"{header}\n{b}" for b in bodies])
    out: list[str] = []
    for body, n in zip(bodies, counts):
        out.extend(_refit(body, n, header, overlap, counter, max_tokens, stats))
    return out


def chunk_sections(
    sections: Sequence[Section],
    *,
    ticker: str,
    form_type: str,
    accession_number: str,
    filing_date: str,
    report_date: str | None = None,
    company_name: str | None = None,
    source_url: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    skip_sections: Sequence[str] = (),
    token_counter: TokenCounter | None = None,
    max_tokens: int | None = None,
    stats: ChunkStats | None = None,
) -> list[Chunk]:
    """Turn sections into Chunks. IDs are deterministic (accession:section:index), so
    re-ingesting the same filing with the same settings replaces rather than duplicates."""
    skip = {s.upper() for s in skip_sections}
    chunks: list[Chunk] = []
    for sec in sections:
        if sec.key.upper() in skip or not sec.paragraphs:
            continue
        header = f"[{ticker} {form_type} | filed {filing_date} | {sec.label}]"
        bodies = _bodies(sec.paragraphs, header, max_chars, overlap, token_counter, max_tokens, stats)
        if len(bodies) > 1:  # drop stray fragments, but keep a section's only chunk
            bodies = [b for b in bodies if len(b) >= MIN_CHUNK_CHARS]  # ("Item 1B: None." is a fact)
        for i, body in enumerate(bodies):
            metadata: dict[str, MetadataValue] = {
                "doc_type": "filing",
                "ticker": ticker,
                "form_type": form_type,
                "accession_number": accession_number,
                "filing_date": filing_date,
                "section": sec.key,
                "section_title": sec.title,
                "chunk_index": i,
            }
            if report_date:
                metadata["report_date"] = report_date
            if company_name:
                metadata["company_name"] = company_name
            if source_url:
                metadata["source_url"] = source_url
            chunks.append(Chunk(f"{accession_number}:{sec.key}:{i:04d}", f"{header}\n{body}", metadata))
    return chunks


def chunk_filing(html: str, **kwargs) -> list[Chunk]:
    """Convenience: HTML -> sections -> chunks. kwargs as chunk_sections()."""
    return chunk_sections(parse_sections(html, kwargs["form_type"]), **kwargs)
