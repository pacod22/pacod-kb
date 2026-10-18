# Evaluation corpus — sources and licences

The files in this directory are the retrieval evaluation corpus used by
`eval/run_eval.py --suite v2`. They were converted from the public sources below by
`build_corpus.py` (`uv run python -m eval.corpus.build_corpus --download-dir DIR`; EUR-Lex
texts are fetched from the Publications Office's Cellar, which serves the same Official Journal
XHTML without EUR-Lex's bot challenge);
`manifest.jsonl` records the source URL, licence and retrieval date of every file.
The corpus is test data only. It is not part of the product.

| File | Source | Licence / status |
|---|---|---|
| `gdpr_en.md`, `gdpr_de.md`, `gdpr_bg.md` | Regulation (EU) 2016/679 (GDPR), articles only (recitals omitted), EUR-Lex | © European Union, https://eur-lex.europa.eu. Reuse authorised under Commission Decision 2011/833/EU. Only the Official Journal text is authentic |
| `led_en.md`, `led_de.md`, `led_bg.md` | Directive (EU) 2016/680 (Law Enforcement Directive), articles only, EUR-Lex | As above |
| `constitution_bg.md` | Constitution of the Republic of Bulgaria, bg.wikisource.org | Official text: not subject to copyright (Bulgarian Copyright and Neighbouring Rights Act, art. 4(1)) |
| `constitution_ru.md` | Constitution of the Russian Federation (edition of 14.03.2020), Section One (articles 1–137), ru.wikisource.org | Official text: not subject to copyright (Civil Code of the Russian Federation, art. 1259(6)) |
| `constitution_hr.md` | Constitution of the Republic of Croatia, hr.wikisource.org | Official text: not subject to copyright (Croatian Copyright and Related Rights Act, art. 8) |
| `constitution_sl.md` | Constitution of the Republic of Slovenia, sl.wikisource.org | Official text: not subject to copyright (Slovenian Copyright and Related Rights Act, art. 9) |
| `wiki_en_photosynthesis.md` | "Photosynthesis", en.wikipedia.org | CC BY-SA 4.0, Wikipedia contributors |
| `wiki_en_plate_tectonics.md` | "Plate tectonics", en.wikipedia.org | CC BY-SA 4.0, Wikipedia contributors |
| `wiki_de_photovoltaik.md` | "Photovoltaik", de.wikipedia.org | CC BY-SA 4.0, Wikipedia contributors |
| `wiki_de_waermepumpe.md` | "Wärmepumpe", de.wikipedia.org | CC BY-SA 4.0, Wikipedia contributors |

The Wikipedia files are adaptations (reference, "see also" and link sections removed, formulas
dropped), so they are distributed under the same licence, CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). The page history at each source URL lists
the authors.

## Golden set

`../golden_v2.jsonl` holds 92 hand-written queries in five buckets:

- `paraphrase`: natural-language questions without the document's wording.
- `exact_term`: article numbers, identifiers, technical terms.
- `cross_lingual`: queries whose answer exists only in another language. Labels never point at a
  translation in the query's language.
- `native`: non-English queries over a same-language document, inflected forms included.
- `negative`: questions with no answer in the corpus. Some are off-topic; others ask about
  the corpus's own subjects, e.g. the European Data Protection Board's headquarters, so a score
  threshold is not calibrated only on easy negatives.

The GDPR and the Law Enforcement Directive share wording and article numbers. Queries about
companies, employers or customers are labelled with the GDPR article only. Queries about police
or criminal matters are labelled with the LED article only. "GDPR Article 30" and "Richtlinie
(EU) 2016/680 Artikel 30" check that an article number goes to the right instrument.

Each non-negative query names its relevant sections and a `must_contain` snippet copied verbatim
from the first one. The harness fails if the snippet is not in that section, so a mislabelled
article number cannot pass silently. Entries sharing a `group` are parallel translations of the
same article: retrieving any of them counts as one hit.
