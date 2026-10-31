# Python compatibility oracle

The oracle uses the original, unchanged Python implementation at commit
`1acefe89412b20245db5a22d2a02001e547dc602`. It refuses to run if any
`minbpe/*.py` source differs from that revision. The fixture data must never be
regenerated using the C# implementation under test.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-oracle.txt
.venv/bin/python scripts/generate-fixtures.py
.venv/bin/python -m pytest -q tests/test_tokenizer.py
```

The initial generation used Python 3.12.3. Dependency versions, reference source
SHA-256 hashes, and upstream revision are embedded in the generated JSON.
Dependencies include transitive packages pinned for repeatability. `tiktoken`
downloads its public `cl100k_base` encoding data on first use; generated C#
resources need no network access at runtime. Synthetic fixture strings are
original test data, except the short established Wikipedia BPE example
`aaabdaaabac`. Existing upstream tests use their own unchanged corpus files.

## Fixture schema

`tests/MinBpe.Compatibility/fixtures.json` contains:

- `cases`: trained basic/regex tokenizers with expected merges, Python `.model`
  and display-only `.vocab` content, registered specials, and query results.
  A null regex pattern selects the upstream default. Named `.model` copies are
  also saved to `tests/MinBpe.Compatibility/models/`.
- `decodeCases`: untrained byte-vocabulary basic/regex decode results, GPT4
  decode results, invalid token errors, and malformed UTF-8 replacement cases.
- `gpt4Cases`: pretrained GPT4 query results, including special-token modes.
  Every ordinary GPT4 query is additionally checked against official tiktoken.
- `splitCases`: regex chunk boundaries for default GPT4, GPT2, and a portable
  custom pattern, including astral Unicode letters and numbers.
- `trainErrorCases`: exhausted training data and invalid vocabulary size errors.

Query `mode` is `none`, `none_raise`, `all`, or `set`; `allowed` holds the names
for set mode. An `error: true` entry means the reference raises, otherwise `ids`
or `text` contains the exact expected result. No exception text/type equivalence
is claimed across languages.

Upstream `load()` restores pattern text but does **not** recompile the regex,
and does **not** populate the inverse special-token map. Its save/load oracle
therefore constructs the matching pattern and re-registers special tokens
before comparing queries. The C# port can repair these implementation quirks;
the fixture documents reference file compatibility rather than preserving bugs.

## Packed pretrained resource

`src/MinBpe/Resources/cl100k_base.bin.gz` is gzip-compressed with zero timestamp.
The decompressed layout is:

1. Eight ASCII magic bytes: `MINBPE01`.
2. Little-endian signed int32 merge count (100000).
3. 256 bytes of `byte_shuffle[i]`, in input byte order.
4. `count` little-endian int32 triples `(left, right, rank)`, in upstream merge
   dictionary iteration order.

The adjacent provenance JSON records source hashes, dependency versions, data
source, dataset hash, and compressed/uncompressed SHA-256 hashes. The generator
checks the canonical public dataset SHA-256 before writing provenance. The
adjacent `TIKTOKEN-LICENSE.txt` preserves OpenAI's MIT attribution for the
pretrained data and tiktoken-derived material.

## Frozen Unicode classes

`src/MinBpe/Resources/unicode-classes.json` freezes full-scalar character
classification, so .NET runtime/ICU upgrades do not change regex splitting or
display-vocabulary output. The generator evaluates every scalar from U+0000
through U+10FFFF, excluding surrogates, and coalesces consecutive matching
scalars into sorted disjoint inclusive ranges. `letter`, `number`, and
`whitespace` come from the pinned `regex` package's `\p{L}`, `\p{N}`, and `\s`.
`control` comes from Python `unicodedata.category(ch).startswith("C")`, matching
upstream `render_token` (this includes unassigned and private-use characters,
not merely C0 controls). The resource records both source versions; provenance
also includes its SHA-256. `UNICODE-LICENSE.txt` is the official Unicode license
retrieved from <https://www.unicode.org/license.txt>; `REGEX-LICENSE.txt` preserves
the installed regex package's license and notices.

Every main generated resource and fixture JSON is written with an atomic
same-directory replacement. Provenance is published last, after all resources
have been generated and validated. Consumers can check the recorded hashes to
detect a generation interrupted between individual file replacements.
