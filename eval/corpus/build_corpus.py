"""Build the multilingual evaluation corpus (eval/corpus/*.md + manifest.jsonl).

The corpus is committed; this script documents and reproduces how it was made.
It downloads public sources into a cache directory (untrusted input: parsed as
data only), converts each into Markdown with explicit headings, and writes the
manifest the harness reads.

Usage:
    uv run python -m eval.corpus.build_corpus --download-dir /tmp/kb-corpus-dl

Sources and licences: see SOURCES.md next to this file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

CORPUS_DIR = Path(__file__).parent
_UA = "pacod-kb-eval/1.0 (retrieval eval corpus builder)"


@dataclass(frozen=True)
class Source:
    file: str
    lang: str
    doc_type: str
    url: str
    license: str
    convert: Callable[[bytes], tuple[str, str]]  # raw bytes -> (title, markdown body)
    fetch_url: str = ""  # differs from url for API-backed sources
    accept_language: str = ""


# ---------------------------------------------------------------------------
# EUR-Lex (GDPR)
# ---------------------------------------------------------------------------


_GDPR_TITLES = {
    "en": "Regulation (EU) 2016/679 — General Data Protection Regulation",
    "de": "Verordnung (EU) 2016/679 — Datenschutz-Grundverordnung",
    "bg": "Регламент (ЕС) 2016/679 — Общ регламент относно защитата на данните",
}
# The Law Enforcement Directive: same domain and wording as the GDPR (erasure, DPO,
# breach notification, adequacy), so the two instruments compete for every query.
_LED_TITLES = {
    "en": "Directive (EU) 2016/680 — Law Enforcement Directive (data protection in criminal "
    "matters)",
    "de": "Richtlinie (EU) 2016/680 — Datenschutz-Richtlinie für Polizei und Justiz",
    "bg": "Директива (ЕС) 2016/680 — защита на данните в наказателноправната област",
}


def _text(el: object) -> str:
    from lxml import html as lhtml

    assert isinstance(el, lhtml.HtmlElement)
    return re.sub(r"\s+", " ", el.text_content()).strip()


def _eurlex(title: str) -> Callable[[bytes], tuple[str, str]]:
    def convert(raw: bytes) -> tuple[str, str]:
        from lxml import html as lhtml

        root = lhtml.fromstring(raw)
        out: list[str] = []
        started = False
        pending_chapter: str | None = None
        art_head: str | None = None
        for el in root.iter("p"):
            cls = el.get("class") or ""
            if cls == "oj-ti-section-1":
                started = True
                pending_chapter = _text(el)
                continue
            if not started:
                continue  # skip the recitals
            if cls == "oj-ti-section-2":
                label = pending_chapter or ""
                heading = f"{label} — {_text(el)}" if label else _text(el)
                level = "###" if re.match(r"^(Section|Abschnitt|Раздел)\b", label) else "##"
                out.append(f"{level} {heading}")
                pending_chapter = None
                continue
            if cls == "oj-ti-art":
                art_head = _text(el)
                continue
            if cls == "oj-sti-art" and art_head:
                out.append(f"#### {art_head} — {_text(el)}")
                art_head = None
                continue
            if cls != "oj-normal" or el.getparent() is None:
                continue
            if el.xpath("ancestor::table"):
                # Enumerations are two-cell table rows: "(a)" | text, nested for
                # sub-points. Emit each row once, from its label cell, with the
                # text cell's own paragraphs (nested rows follow on their own).
                td = el.getparent()
                tr = td.getparent() if td is not None and td.tag == "td" else None
                if tr is None:
                    continue
                tds = [c for c in tr if c.tag == "td"]
                if td is not tds[0]:
                    continue
                rest = " ".join(_text(p) for p in tds[1].findall("p")) if len(tds) > 1 else ""
                row = f"{_text(el)} {rest}".strip()
                if row:
                    out.append(row)
                continue
            if el.xpath(
                "ancestor::*[contains(@class,'oj-final') or contains(@class,'oj-signatory')]"
            ):
                continue
            t = _text(el)
            if t:
                out.append(t)
        body = "\n\n".join(_dedupe_adjacent(out))
        return title, body

    return convert


def _dedupe_adjacent(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for it in items:
        if not (out and out[-1] == it):
            out.append(it)
    return out


# ---------------------------------------------------------------------------
# MediaWiki plain-text extracts (Wikisource, Wikipedia)
# ---------------------------------------------------------------------------


_DROP_SECTIONS = {
    "references", "see also", "external links", "notes", "further reading",
    "bibliography", "sources", "citations", "literatur", "weblinks",
    "einzelnachweise", "siehe auch", "anmerkungen", "quellen",
}


def _extract(raw: bytes) -> str:
    data = json.loads(raw)
    page = next(iter(data["query"]["pages"].values()))
    return str(page.get("extract", ""))


def _heading(line: str) -> tuple[int, str] | None:
    m = re.match(r"^(=+)\s*(.*?)\s*=+\s*$", line)
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


def _wikipedia(title: str) -> Callable[[bytes], tuple[str, str]]:
    def convert(raw: bytes) -> tuple[str, str]:
        out: list[str] = []
        skip_level: int | None = None
        for line in _extract(raw).splitlines():
            line = line.strip()
            # Formulas come out as one symbol per line ("C", "O", "P", "=").
            if len(line) <= 3 or "{\\displaystyle" in line:
                continue
            h = _heading(line)
            if h:
                level, text = h
                if skip_level is not None and level > skip_level:
                    continue
                skip_level = level if text.casefold() in _DROP_SECTIONS else None
                if skip_level is None:
                    out.append(f"{'#' * min(level, 6)} {text}")
                continue
            if skip_level is None:
                out.append(line)
        return title, "\n\n".join(out)

    return convert


def _bg_constitution(raw: bytes) -> tuple[str, str]:
    out: list[str] = []
    for line in _extract(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        h = _heading(line)
        if h:
            out.append(f"## {h[1]}")
            continue
        m = re.match(r"^(Чл\. ?\d+[а-я]?)\.\s*(.*)$", line)
        if m:
            out.append(f"#### {m.group(1).replace('Чл.', 'Чл. ').replace('  ', ' ')}")
            if m.group(2):
                out.append(m.group(2))
            continue
        if out:  # drop the publication notes before the first heading
            out.append(line)
    return "Конституция на Република България", "\n\n".join(out)


def _hr_constitution(raw: bytes) -> tuple[str, str]:
    out: list[str] = []
    for line in _extract(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        h = _heading(line)
        if h:
            level, text = h
            out.append(f"#### {text}" if text.startswith("Članak") else f"## {text}")
            continue
        out.append(line)
    return "Ustav Republike Hrvatske", "\n\n".join(out)


def _sl_constitution(raw: bytes) -> tuple[str, str]:
    out: list[str] = []
    for line in _extract(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        h = _heading(line)
        if h:
            level, text = h
            out.append(f"{'#' * min(level, 3)} {text}")
            continue
        if re.match(r"^\d+[a-z]?\. člen(\s*\(.*\))?$", line):
            out.append(f"#### {line}")
            continue
        out.append(line)
    return "Ustava Republike Slovenije", "\n\n".join(out)


# ---------------------------------------------------------------------------
# Russian Wikisource raw wikitext (the extracts API drops its templates)
# ---------------------------------------------------------------------------


def _clean_wikitext(line: str) -> str:
    # Inserted articles/parts are superscripts ("Статья 67<sup>1*</sup>"): 67.1
    line = re.sub(r"<sup>\s*(\d+)\s*\*?\s*</sup>", r".\1", line)
    line = re.sub(r"<sup>\s*\*\s*</sup>", "", line)
    line = re.sub(r"<ref[^>]*/>", "", line)
    line = re.sub(r"<ref[^>]*>.*?</ref>", "", line)
    line = re.sub(r"\{\{якорь\|[^}]*\}\}", "", line)
    line = re.sub(r"\{\{(?:left|center|right|razr|nobr)\|([^{}]*)\}\}", r"\1", line)
    line = re.sub(r"\{\{[^{}]*\}\}", "", line)
    line = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", line)
    line = re.sub(r"\[\[([^\]]*)\]\]", r"\1", line)
    line = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", line)
    line = re.sub(r"</?br\s*/?>", " ", line)
    line = re.sub(r"<[^>]+>", "", line)
    line = line.replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", line).strip()


def _ru_constitution(max_article: int) -> Callable[[bytes], tuple[str, str]]:
    def convert(raw: bytes) -> tuple[str, str]:
        out: list[str] = []
        started = False
        for line in raw.decode("utf-8").splitlines():
            h = _heading(line.strip())
            if h:
                started = True
                level, text = h
                text = _clean_wikitext(text)
                m = re.match(r"^Статья (\d+)", text)
                if m and int(m.group(1)) > max_article:
                    break
                if text.startswith("РАЗДЕЛ ВТОРОЙ"):
                    break
                out.append(f"{'#' * min(level, 4)} {text}")
                continue
            if not started:
                continue
            t = _clean_wikitext(line)
            if t:
                out.append(t)
        return "Конституция Российской Федерации", "\n\n".join(out)

    return convert


# ---------------------------------------------------------------------------


def _mw(host: str, title: str) -> str:
    from urllib.parse import urlencode

    q = urlencode(
        {
            "action": "query", "prop": "extracts", "explaintext": "1",
            "exsectionformat": "wiki", "redirects": "1", "format": "json", "titles": title,
        }
    )
    return f"https://{host}/w/api.php?{q}"


_ISO639_2 = {"en": "eng", "de": "deu", "bg": "bul"}
_EUR_LICENSE = (
    "© European Union, https://eur-lex.europa.eu — reuse authorised (Decision 2011/833/EU)"
)
_CC = "CC BY-SA 4.0 (Wikipedia contributors)"

SOURCES: list[Source] = [
    *[
        Source(
            file=f"{name}_{lang}.md", lang=lang, doc_type="legal_code",
            url=f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/HTML/?uri=CELEX:{celex}",
            # EUR-Lex answers scripted clients with a bot challenge; the Publications
            # Office's Cellar serves the same Official Journal XHTML.
            fetch_url=f"https://publications.europa.eu/resource/celex/{celex}",
            accept_language=_ISO639_2[lang],
            license=_EUR_LICENSE, convert=_eurlex(titles[lang]),
        )
        for name, celex, titles in (
            ("gdpr", "32016R0679", _GDPR_TITLES),
            ("led", "32016L0680", _LED_TITLES),
        )
        for lang in ("en", "de", "bg")
    ],
    Source(
        file="constitution_bg.md", lang="bg", doc_type="legal_code",
        url="https://bg.wikisource.org/wiki/Конституция_на_Република_България",
        fetch_url=_mw("bg.wikisource.org", "Конституция на Република България"),
        license="Official text, not subject to copyright (Bulgarian Copyright Act, art. 4(1))",
        convert=_bg_constitution,
    ),
    Source(
        file="constitution_ru.md", lang="ru", doc_type="legal_code",
        url="https://ru.wikisource.org/wiki/Конституция_Российской_Федерации/Редакция_от_14.03.2020",
        fetch_url=(
            "https://ru.wikisource.org/w/index.php?action=raw&title="
            "Конституция_Российской_Федерации/Редакция_от_14.03.2020"
        ),
        license="Official text, not subject to copyright (Civil Code of the RF, art. 1259(6))",
        convert=_ru_constitution(max_article=137),
    ),
    Source(
        file="constitution_hr.md", lang="hr", doc_type="legal_code",
        url="https://hr.wikisource.org/wiki/Ustav_Republike_Hrvatske",
        fetch_url=_mw("hr.wikisource.org", "Ustav Republike Hrvatske"),
        license="Official text, not subject to copyright (Croatian Copyright Act, art. 8)",
        convert=_hr_constitution,
    ),
    Source(
        file="constitution_sl.md", lang="sl", doc_type="legal_code",
        url="https://sl.wikisource.org/wiki/Ustava_Republike_Slovenije",
        fetch_url=_mw("sl.wikisource.org", "Ustava Republike Slovenije"),
        license="Official text, not subject to copyright (Slovenian Copyright Act ZASP, art. 9)",
        convert=_sl_constitution,
    ),
    Source(
        file="wiki_en_photosynthesis.md", lang="en", doc_type="article",
        url="https://en.wikipedia.org/wiki/Photosynthesis",
        fetch_url=_mw("en.wikipedia.org", "Photosynthesis"), license=_CC,
        convert=_wikipedia("Photosynthesis"),
    ),
    Source(
        file="wiki_en_plate_tectonics.md", lang="en", doc_type="article",
        url="https://en.wikipedia.org/wiki/Plate_tectonics",
        fetch_url=_mw("en.wikipedia.org", "Plate tectonics"), license=_CC,
        convert=_wikipedia("Plate tectonics"),
    ),
    Source(
        file="wiki_de_photovoltaik.md", lang="de", doc_type="article",
        url="https://de.wikipedia.org/wiki/Photovoltaik",
        fetch_url=_mw("de.wikipedia.org", "Photovoltaik"), license=_CC,
        convert=_wikipedia("Photovoltaik"),
    ),
    Source(
        file="wiki_de_waermepumpe.md", lang="de", doc_type="article",
        url="https://de.wikipedia.org/wiki/Wärmepumpe",
        fetch_url=_mw("de.wikipedia.org", "Wärmepumpe"), license=_CC,
        convert=_wikipedia("Wärmepumpe"),
    ),
]


def _download(src: Source, dl: Path) -> bytes:
    import httpx

    target = dl / (src.file + ".raw")
    if target.exists():
        return target.read_bytes()
    headers = {"User-Agent": _UA}
    if src.accept_language:
        headers |= {"Accept": "application/xhtml+xml", "Accept-Language": src.accept_language}
    resp = httpx.get(src.fetch_url or src.url, headers=headers, timeout=90, follow_redirects=True)
    resp.raise_for_status()
    if resp.status_code != 200 or not resp.content:
        # e.g. a WAF challenge (202, empty body): never cache it as the source.
        raise RuntimeError(f"{src.file}: HTTP {resp.status_code}, {len(resp.content)} bytes")
    target.write_bytes(resp.content)
    return resp.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--retrieved", default=date.today().isoformat())
    args = parser.parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    for src in SOURCES:
        title, body = src.convert(_download(src, args.download_dir))
        text = f"# {title}\n\n{body.strip()}\n"
        (CORPUS_DIR / src.file).write_text(text, encoding="utf-8")
        manifest.append(
            {
                "file": src.file, "doc_type": src.doc_type, "lang": src.lang, "title": title,
                "source_url": src.url, "license": src.license, "retrieved": args.retrieved,
            }
        )
        print(f"{src.file:32} {len(text.encode()):>9,} bytes", file=sys.stderr)
    with (CORPUS_DIR / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in manifest:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
