"""Query analysis for the keyword leg: stopwords, light stemming, legal references.

The keyword indexes (SQLite FTS5 ``unicode61``, PostgreSQL ``simple``) do no stemming
and keep every word. Requiring every query word (the legacy AND) makes natural
questions match nothing ("What does Article 17 say?" needs "what" and "does"). Here
the query side drops function words, turns inflected words into prefixes (законите →
закон*, personenbezogener → personenbezogen*) and keeps article references as phrases
("чл 17"), so the same unchanged indexes can be searched by recall: any term may match,
and BM25 ranks chunks that match more and rarer terms first.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

# Function words and question words. Domain words that carry meaning in legal text
# (article, law, court, ...) are deliberately absent.
_STOP_LATIN = frozenset(
    # English
    "a about above after again against all am an and any are as at be because been before "
    "being below between both but by can could did do does doing down during each few for "
    "from further had has have having he her here hers herself him himself his how i if in "
    "into is it its itself just me more most my myself no nor not now of off on once only or "
    "other our ours ourselves out over own same she should so some such than that the their "
    "theirs them themselves then there these they this those through to too under until up "
    "very was we were what when where which while who whom why will with would you your yours "
    "yourself yourselves may might must shall also whether s t "
    # German
    "aber alle allem allen aller alles als also am an ander andere anderem anderen anderer "
    "anderes auch auf aus bei bin bis bist da damit dann das dass dein deine dem den der des "
    "dessen dich die dies diese diesem diesen dieser dieses dir doch dort du durch ein eine "
    "einem einen einer eines er es etwas euch euer für gegen hab habe haben hat hatte hier hin "
    "hinter ich ihm ihn ihnen ihr ihre im in ist ja jede jedem jeden jeder jedes jetzt kann "
    "kein keine können könnte machen man mein meine mich mir mit muss musste nach nicht nichts "
    "noch nun nur ob oder ohne sehr sein seine sich sie sind so solche soll sollte sondern "
    "über um und uns unser unter vom von vor war waren warum was weil welche welcher welches "
    "wenn wer werde werden wie wieder will wir wird wo wollen zu zum zur zwischen wieviel wie "
    "viele vieler gibt darf dürfen "
    # Croatian, Serbian (Latin), Slovenian
    "i u na je se da za od do iz s sa po o ali ili a kao što koji koja koje kojih te to taj "
    "ta ti su biti bi bio bila bilo smo ste sam si li ne ni pa kad kada gdje gde kako zašto "
    "koliko ima nema može mogu ovaj ova ovo oni one ona on in v na pri za so ki ali kot tudi "
    "še že ter bo bil bila bilo sem si smo ste so je da ne se kje kdaj kako zakaj koliko kaj "
    "kdo ta to tisti tista tisto".split()
)
_STOP_CYRILLIC = frozenset(
    # Bulgarian
    "а аз ако ала би бил била били било в във вие всички все всеки всяка всяко го да дали до "
    "ако е едва ето за защо зад и из или им има имат иска й каза как каква какво какъв като "
    "кога когато кое кои кой който колко която което които към ли ме между мен ми много може "
    "мога могат моля му на над нас не него нея ни ние никой нито но някой някои няма о обаче "
    "около от отново пак по пред при пък с са само се си след сме сте със също та така там те "
    "тези ти то това тогава този той три тук тя тях у че чрез ще щом я "
    # Russian
    "а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже "
    "для до его ее ей ему если есть еще же за здесь и из или им их к как какая какой какие "
    "когда кто ли либо мне может мы на над надо наш не него нее нет ни них но ну о об один он "
    "она они оно от очень по под при с со так также такой там те тем то того тоже той только "
    "том ты у уж уже хотя чего чей чем что чтобы чье чья эта эти это этот я сколько какое "
    "какова каков каковы ли нужно можно "
    # Macedonian, Serbian (Cyrillic)
    "и во на е се да за од до со по ги го ја не ни но што кој која кое кои како кога каде зошто "
    "колку има нема може или или је су у из са као што који која које када где зашто колико".split()
)
# Never dropped, even when short: article and paragraph abbreviations.
_KEEP = frozenset({"art", "abs", "nr", "чл", "ал", "ст", "т", "član", "čl", "§"})
# A marker followed by a number becomes a phrase: "чл. 17" -> "чл 17".
_LEGAL_MARKERS = frozenset(
    {
        "article", "articles", "art", "artikel", "artikels", "abs", "absatz", "paragraph", "чл",
        "член", "члена", "ал", "алинея", "статья", "статьи", "статье", "статьей", "ст", "часть",
        "član", "člana", "članak", "članka", "čl", "člen", "člena", "členu", "stav", "odstavek",
    }
)

# Inflection endings, longest first. A word keeps at least 4 letters, then matches as a
# prefix: Slavic case/number/article endings are 1-3 letters, so a fixed "drop two
# letters" rule would cut short words too far and long endings too little.
_CYRILLIC_SUFFIXES = sorted(
    (
        "иями ями ами ими ыми ого его ому ему ите ата ото ове ища ият ия ья ье ию ью ая яя ое ее "
        "ые ие ых их ым им ой ей ом ем ам ям ах ях ов ев ът ят ый ий ую юю ою ею та то те ти а я "
        "о е и ы у ю ь й"
    ).split(),
    key=len,
    reverse=True,
)
_LATIN_SUFFIXES = sorted(
    (
        "ovima evima ama ima ing ung ed es en er em om ih im oj ega emu ih "
        "ov ev ja je ju e n s a i o u"
    ).split(),
    key=len,
    reverse=True,
)
_MIN_STEM = 4
_MIN_PREFIX = 5


@dataclass(frozen=True)
class QueryTerm:
    text: str
    prefix: bool


@dataclass(frozen=True)
class KeywordQuery:
    terms: tuple[QueryTerm, ...]
    phrases: tuple[tuple[str, str], ...]

    @property
    def empty(self) -> bool:
        return not self.terms and not self.phrases


def light_stem(token: str) -> tuple[str, bool]:
    """(stem, match_as_prefix) for a lower-cased word."""
    suffixes = _CYRILLIC_SUFFIXES if _CYRILLIC_RE.search(token) else _LATIN_SUFFIXES
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            return token[: -len(suffix)], True
    return token, len(token) >= _MIN_PREFIX


def _is_stopword(token: str) -> bool:
    if token in _KEEP:
        return False
    return token in (_STOP_CYRILLIC if _CYRILLIC_RE.search(token) else _STOP_LATIN)


def analyze_query(query: str, *, max_terms: int = 16) -> KeywordQuery:
    """Terms and article phrases for the keyword leg (see the module docstring).

    Lower-cased with str.lower(), not casefold(): the index tokenizers lower-case, and
    casefold would turn "ß" into "ss", which the index never contains.
    """
    raw = _TOKEN_RE.findall(unicodedata.normalize("NFC", query))
    lowered = [t.lower() for t in raw]
    terms: list[QueryTerm] = []
    phrases: list[tuple[str, str]] = []
    seen: set[QueryTerm] = set()

    def add(term: QueryTerm) -> None:
        if term not in seen:
            seen.add(term)
            terms.append(term)

    for i, (original, token) in enumerate(zip(raw, lowered, strict=True)):
        nxt = lowered[i + 1] if i + 1 < len(lowered) else ""
        if token in _LEGAL_MARKERS and any(c.isdigit() for c in nxt):
            phrase = (token, nxt)
            if phrase not in phrases:
                phrases.append(phrase)
        acronym = original.isupper() and 2 <= len(original) <= 6  # "IT", "US", "DSGVO"
        if not acronym and _is_stopword(token):
            continue
        exact = acronym or any(c.isdigit() for c in token) or len(token) <= 3
        if exact:
            add(QueryTerm(token, prefix=False))
        else:
            stem, prefix = light_stem(token)
            add(QueryTerm(stem, prefix))
        if "ё" in token:
            stem, prefix = light_stem(token.replace("ё", "е"))
            add(QueryTerm(stem, prefix))
    return KeywordQuery(tuple(terms[:max_terms]), tuple(phrases))
