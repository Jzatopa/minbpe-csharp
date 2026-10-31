# Compatibility contract

## Pinned reference

- Source: https://github.com/karpathy/minbpe
- Commit: `1acefe89412b20245db5a22d2a02001e547dc602`
- Original `minbpe/*.py` files are unchanged. Oracle generation refuses to run if their SHA-256 hashes differ.
- Python, regex, tiktoken and pytest versions are pinned/recorded in `scripts/requirements-oracle.txt` and fixture metadata.
- Pretrained data is exactly the pinned `cl100k_base` encoding. Its canonical data hash, compressed/uncompressed resource hashes, source hashes and Unicode snapshot provenance are in `src/MinBpe/Resources/cl100k_base.provenance.json`.

## API mapping

| Python | C# |
| --- | --- |
| `BasicTokenizer`, `RegexTokenizer`, `GPT4Tokenizer` | Same names, namespace `MinBpe` |
| `train(text, vocab_size, verbose=False)` | `Train(text, vocabularySize, verbose: false)` |
| `encode`, `decode` | `Encode` → `int[]`; `Decode` → `string` |
| `encode_ordinary` | `RegexTokenizer.EncodeOrdinary` |
| `register_special_tokens` | `RegisterSpecialTokens` using ordered key/value entries |
| `allowed_special="none_raise"` | Default `Encode(text)` / `SpecialTokenMode.NoneRaise` |
| `"none"`, `"all"`, or a set | `SpecialTokenMode.None`, `All`, or an `IReadOnlySet<string>` |
| `save`, `load` | `Save(prefix)` and `Load(path.model)` |
| `save_vocab` (GPT4) | `SaveVocab(path)`; also available for trainable tokenizers |
| `merges`, `vocab` inspection | `Merges`, `VocabularySize`, `GetTokenBytes(id)` |

## Preserved semantics

- Training counts overlapping pairs, merges non-overlapping occurrences left-to-right, and resolves equal-frequency ties by the pair's first occurrence in the current chunk sequence. Chunk boundaries are never crossed by a training merge.
- Encoding repeatedly selects the available pair with the lowest learned rank and merges all its occurrences. No normalization or case folding is applied to input text outside the source's contraction-matching alternative.
- Built-in GPT2/GPT4 splitting is implemented over Unicode scalars, not UTF-16 code units. Unicode letter/number/whitespace classes are exported from pinned Python `regex`; vocabulary-rendering C* classes come from pinned Python `unicodedata`. Runtime Unicode upgrades do not silently change these tables; contraction case matching still uses the runtime’s invariant case conversion.
- The splitter preserves ordered alternatives, GPT4's possessive-prefix/punctuation behavior, one-to-three-scalar numeric groups, contraction case policy and whitespace backtracking.
- Specials use ordinal strings, earliest position and registration-order precedence, not longest-match precedence. Unregistered names in an explicit allow-set are ignored; registered names not allowed by that set are encoded ordinarily, not rejected. `NoneRaise` rejects every registered special string in input.
- UTF-8 bytes are concatenated before replacement decoding, including across token boundaries. Ill-formed UTF-16 input is rejected instead of silently replacing lone surrogates during encoding.
- Models use `minbpe v1`, UTF-8 without BOM and LF newlines, with sequential merge ranks reconstructed from line order. `.vocab` is display-only and lossy, never a loadable model.
- As upstream, a vocabulary larger than the training text can sustain raises when no pairs remain; the implementation does not invent merges or silently shrink the requested vocabulary.
- Pretrained byte shuffle/inverse shuffle and 100,000 learned merges are retained. `GPT4Tokenizer.Train`, `Save`, and `Load` are unsupported, as upstream; `SaveVocab` unshuffles bytes for human inspection.
- Upstream's GPT4 decoder does not recognize special IDs; the C# port likewise raises for these IDs, even though its encoder can emit them when explicitly allowed. It is not a claim to support every behavior of tiktoken's public API.

## Deliberate fixes and boundaries

- `Load` restores the saved splitting pattern and special-token decoding immediately. The Python source restores pattern text without recompiling its regex. Its loaded vocabulary normally supplies special decoding despite the stale inverse map. C# restores consistent pattern/special state; the oracle constructs the matching pattern for comparison.
- Re-registering specials after a load discards stale special bytes from the loaded vocabulary, preventing obsolete ID-to-text mappings.
- Specials must be nonempty, whitespace-free strings with distinct nonnegative IDs outside the ordinary vocabulary; collisions and invalid registrations are rejected instead of reproducing Python's ambiguous states. Reserve enough ID space before later training.
- `VocabularySize` always counts the normal byte/merge vocabulary, excluding special IDs; `SpecialTokens.Count` is separate. Use a fresh BasicTokenizer to retrain if special IDs imported from another model constrain the requested vocabulary.
- Models with invalid headers, duplicate/out-of-order merges or invalid special entries are rejected. Failed training/loading does not partially overwrite the current tokenizer. Save rejects multi-line or leading/trailing-whitespace patterns that minbpe v1 cannot round-trip. Load rejects ambiguous whitespace patterns, a UTF-8 BOM, and empty patterns for RegexTokenizer instead of silently changing encoding behavior. Save is not a transaction across both files: after an I/O failure, discard/recreate the pair or restore a backup.
- Custom patterns use .NET regex syntax, culture-invariant matching and a two-second regex timeout. Capturing groups are rejected because Python `findall` changes its return shape for captures. Arbitrary Python-only regex syntax, Unicode behavior and flags are not portable. Custom `\p{L}`/`\p{N}` classes do not recognize astral letters/numbers, and custom dot/classes that split a surrogate pair are rejected with an explicit error. Use the built-in patterns for scalar-aware behavior. The two built-in constants do not use the .NET regex engine and have their own pinned scalar semantics.
- Exception classes, diagnostic wording, verbose training output and Python's directly mutable internal attributes are not ABI-compatible APIs. `GetTokenBytes` returns a copy; inspection cannot mutate the learned byte vocabulary.
- Trainable instances are mutable. Do not call `Train`, `Load`, or `RegisterSpecialTokens` concurrently with encoding/decoding. Once state is fixed, independent encode/decode calls use per-call buffers.
- `.vocab` follows source state: initial train/register omits specials from the byte vocabulary; load includes them; re-register removes stale loaded special entries. Model/encoding compatibility matters more than display-file idempotence.

## Verification evidence

The checked-in oracle contains 134 trained Basic/Regex models and 1,268 query scenarios exercised before and after loading (2,536 checks), plus 170 pretrained queries: **2,706 encode comparisons**. It also contains 137 decode cases, 113 split cases and 16 expected training failures. Successful models compare complete merge order and exact Python `.model`/`.vocab` output; unique file names avoid interference with original Python tests. The native harness validates source hashes, resource hashes and expected fixture counts before checking behavior.

The fixtures cover default/GPT2/custom patterns, first-seen merge ties, repeated pairs, Unicode beyond BMP, accents, whitespace/CRLF, special allow-modes and overlaps, malformed UTF-8 replacement, pretrained byte shuffling and Python/C# model-file exchange for the built-in patterns and the portable custom `\S+|\s+` fixture pattern; other custom patterns need their own cross-engine tests. Native API checks add invalid-model handling, surrogate rejection and pretrained-operation restrictions. Original upstream pytest supplies 21 tests.

This is bounded compatibility evidence, not proof of every possible text or arbitrary custom regex. No throughput, memory or model-quality improvement is claimed without a separate benchmark/evaluation.
