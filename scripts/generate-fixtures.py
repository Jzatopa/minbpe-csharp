#!/usr/bin/env python3
"""Generate compatibility oracles from unchanged upstream Python, never C# outputs.

Run from any directory: .venv/bin/python scripts/generate-fixtures.py
Models use short original synthetic text; no external corpus is downloaded.
"""
import gzip
import hashlib
import importlib.metadata
import json
import platform
import random
import struct
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import regex
import tiktoken
from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer
from minbpe.regex import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN

COMMIT = "1acefe89412b20245db5a22d2a02001e547dc602"
OUT = ROOT / "tests/MinBpe.Compatibility"
RESOURCES = ROOT / "src/MinBpe/Resources"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    """Replace a complete generated file; never expose a partially written file."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as temporary:
        temporary.write(data)
        name = temporary.name
    Path(name).replace(path)


def unicode_classes():
    """Freeze the Python oracle's Unicode semantics, independent of .NET ICU."""
    predicates = {
        "letter": regex.compile(r"\A\p{L}\Z").fullmatch,
        "number": regex.compile(r"\A\p{N}\Z").fullmatch,
        "whitespace": regex.compile(r"\A\s\Z").fullmatch,
        "control": lambda ch: unicodedata.category(ch).startswith("C"),
    }
    classes = {}
    for name, predicate in predicates.items():
        ranges = []
        for scalar in range(0x110000):
            if 0xD800 <= scalar <= 0xDFFF:
                continue  # No surrogate can occur in valid decoded UTF-8.
            if predicate(chr(scalar)):
                if ranges and ranges[-1][1] == scalar - 1:
                    ranges[-1][1] = scalar
                else:
                    ranges.append([scalar, scalar])
        classes[name] = ranges
    result = {"regexVersion": importlib.metadata.version("regex"),
              "unicodedataVersion": unicodedata.unidata_version, "classes": classes}
    return json.dumps(result, separators=(",", ":")).encode("utf-8") + b"\n"


def query(tokenizer, text, mode="none_raise", allowed=None):
    item = {"text": text, "mode": mode, "allowed": allowed or []}
    try:
        ids = (tokenizer.encode(text) if isinstance(tokenizer, BasicTokenizer)
               else tokenizer.encode(text, set(allowed or []) if mode == "set" else mode))
        item["ids"] = ids
    except (AssertionError, ValueError, KeyError):
        item["error"] = True
    return item


def main():
    # Refuse to silently generate an oracle from modified reference implementation.
    source_hashes = {}
    for path in sorted((ROOT / "minbpe").glob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        expected = subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"], cwd=ROOT)
        actual = path.read_bytes()
        if actual != expected:
            raise RuntimeError(f"Reference source differs from pinned commit: {relative}")
        source_hashes[relative] = sha(actual)
    versions = {p: importlib.metadata.version(p) for p in ("regex", "tiktoken", "pytest")}
    versions["python"] = platform.python_version()
    result = dict(sourceCommit=COMMIT, sourceHashes=source_hashes, versions=versions,
                  cases=[], decodeCases=[], gpt4Cases=[], splitCases=[], trainErrorCases=[])
    rng = random.Random(20261031)
    texts = [
        "aaabdaaabac", "ababcdcdababcdcd", "aaaaaaa", "banana bandana banana",
        "A café! naïve coöperate; e\u0301, Ångström.", "你好世界你好世界 한국어 日本語",
        "𐐀𐐨𐐁𐐩 𝟙𝟚𝟛𝟜 𐒠𐒡𐒢 𞤀𞤢", "a  b\t c\n d\r\n e   ",
        "I'm I'M she'll SHE'LL we've WE'VE can't CAN'T", "1234567890 １２３４５６７８９０",
        "Emoji 👩‍💻 🇺🇸 👍🏽 🌍; hi!?!\r\n", "First line.\r\nSecond line!\n\nFinal.  ",
        "", "x", "\x00\x01\x1f\x7f\u200b\ufeff",
    ]
    alphabet = list("abcde 123!?\n\t") + ["é", "中", "😀", "𐐀", "𝟚", "\r"]
    texts += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(45, 100))) for _ in range(18)]
    common_queries = ["", "hello world", "aaabdaaabac", "naïve café e\u0301", "𐐀 𝟙𐒠😀", "\r\n\t   ", "<x><xy><|end|>"]
    patterns = [("basic", None), ("regex", None), ("regex", GPT2_SPLIT_PATTERN),
                ("regex", r"\S+|\s+")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        for kind, pattern in patterns:
            for i, text in enumerate(texts):
                # Zero-merge models exercise raw-byte behavior; otherwise probe up to 12 merges.
                merges = 0 if i in (12, 13, 14) else (3 if i == 0 else (i % 12) + 1)
                tokenizer = BasicTokenizer() if kind == "basic" else RegexTokenizer(pattern)
                try:
                    tokenizer.train(text, 256 + merges)
                except ValueError:
                    # Keep the largest feasible model; failure is separately captured below.
                    while merges:
                        merges -= 1
                        try:
                            tokenizer.train(text, 256 + merges)
                            break
                        except ValueError:
                            pass
                specials = ({"<x>": 90001, "<xy>": 90002, "<|end|>": 90003}
                            if kind == "regex" and i % 2 == 0 else {})
                if kind == "regex":
                    tokenizer.register_special_tokens(specials)
                name = f"{kind}-{patterns.index((kind, pattern))}-{i:02d}"
                prefix = str(Path(temporary) / name)
                tokenizer.save(prefix)
                model = Path(prefix + ".model").read_text(encoding="utf-8")
                vocab = Path(prefix + ".vocab").read_text(encoding="utf-8")
                (OUT / "models" / f"{name}.model").write_text(model, encoding="utf-8")
                # Verify upstream save/load query parity (custom pattern constructor is intentional:
                # upstream load does not recompile compiled_pattern).
                loaded = BasicTokenizer() if kind == "basic" else RegexTokenizer(pattern)
                loaded.load(prefix + ".model")
                if kind == "regex":
                    loaded.register_special_tokens(specials)
                queries = [query(tokenizer, s) for s in [text] + common_queries]
                if specials:
                    for mode in ("none", "none_raise", "all", "set"):
                        queries.append(query(tokenizer, "before<x><xy><|end|>after", mode, ["<x>"]))
                for item in queries:
                    assert query(loaded, item["text"], item["mode"], item["allowed"]) == item
                result["cases"].append(dict(name=name, kind=kind, pattern=pattern, trainText=text,
                    vocabSize=256 + merges, merges=[[a, b, rank] for (a, b), rank in tokenizer.merges.items()],
                    model=model, vocab=vocab, specials=specials, queries=queries))

        # Prefix overlap tests must preserve registration order, not longest-match order.
        for specials in ({"<a": 91001, "<abc>": 91002}, {"<abc>": 91002, "<a": 91001}):
            tokenizer = RegexTokenizer()
            tokenizer.register_special_tokens(specials)
            name = f"overlap-{len(result['cases'])}"
            prefix = str(Path(temporary) / name)
            tokenizer.save(prefix)
            model = Path(prefix + ".model").read_text()
            (OUT / "models" / f"{name}.model").write_text(model)
            result["cases"].append(dict(name=name, kind="regex", pattern=None, trainText="", vocabSize=256,
                merges=[], model=model, vocab=Path(prefix + ".vocab").read_text(), specials=specials,
                queries=[query(tokenizer, "a<abc>b", mode, ["<abc>"]) for mode in ("all", "set", "none", "none_raise")]))

    for kind, pattern in patterns:
        for text, size in [("", 257), ("a", 257), ("ab", 259), ("abc", 255)]:
            tokenizer = BasicTokenizer() if kind == "basic" else RegexTokenizer(pattern)
            try:
                tokenizer.train(text, size)
            except (AssertionError, ValueError):
                result["trainErrorCases"].append(dict(kind=kind, pattern=pattern, text=text, vocabSize=size))
            else:
                raise AssertionError("Expected reference training failure")

    for pattern in (None, GPT2_SPLIT_PATTERN, r"\S+|\s+"):
        for text in texts + ["A\r\nB\nC\rD", "a  \n \r\n\t", "𐐀𐐨 𐒠𐒡 𝟘𝟙𝟚𝟛", "x's X'S I'LL I'M"]:
            result["splitCases"].append(dict(pattern=pattern, text=text,
                chunks=regex.findall(GPT4_SPLIT_PATTERN if pattern is None else pattern, text)))

    print("Recovering GPT4 merge forest...", flush=True)
    gpt = GPT4Tokenizer()
    enc = tiktoken.get_encoding("cl100k_base")
    for text in texts + common_queries + ["".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 160))) for _ in range(110)]:
        item = query(gpt, text)
        assert item["ids"] == enc.encode(text)
        result["gpt4Cases"].append(item)
    for special in gpt.special_tokens:
        for mode in ("none", "none_raise", "all", "set"):
            result["gpt4Cases"].append(query(gpt, "before" + special + "after", mode, [special]))
    for kind, tokenizer in [("basic", BasicTokenizer()), ("regex", RegexTokenizer()), ("gpt4", gpt)]:
        lists = [[], [0], [255], [195, 169], [240, 159, 152, 128], [192, 175], [237, 160, 128],
                 [226, 130], [226, 130, 65], [128, 128], [256], [-1], [999999], [100257]]
        lists += [[rng.randrange(256) for _ in range(rng.randrange(1, 30))] for _ in range(30)]
        if kind == "gpt4":
            lists += [[rank] for rank in (256, 1000, 100255, 100258, 100276)]
        for ids in lists:
            item = dict(kind=kind, ids=ids)
            try:
                item["text"] = tokenizer.decode(ids)
            except (KeyError, ValueError):
                item["error"] = True
            result["decodeCases"].append(item)

    RESOURCES.mkdir(parents=True, exist_ok=True)
    packed = b"MINBPE01" + struct.pack("<i", len(gpt.merges))
    packed += bytes(gpt.byte_shuffle[i] for i in range(256))
    packed += b"".join(struct.pack("<iii", a, b, rank) for (a, b), rank in gpt.merges.items())
    compressed = gzip.compress(packed, compresslevel=9, mtime=0)
    # Canonical public tiktoken serialization: base64(token) + space + decimal rank + newline.
    import base64
    rank_data = b"".join(base64.b64encode(token) + b" " + str(rank).encode() + b"\n"
                         for token, rank in enc._mergeable_ranks.items())
    assert sha(rank_data) == "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7", "Unexpected cl100k_base dataset"
    unicode_data = unicode_classes()
    with tempfile.TemporaryDirectory() as temporary:
        vocab_path = Path(temporary) / "gpt4.vocab"
        gpt.save_vocab(str(vocab_path))
        vocab_hash = sha(vocab_path.read_bytes())
    provenance = dict(sourceCommit=COMMIT, sourceHashes=source_hashes, versions=versions,
        format="MINBPE01; int32-le merge count; 256-byte shuffle; int32-le (left,right,rank) triples",
        gpt4VocabSha256=vocab_hash, mergeCount=len(gpt.merges), compressedSha256=sha(compressed), uncompressedSha256=sha(packed),
        cl100kRankSha256=sha(rank_data), rankHashSerialization="tiktoken base64-token space decimal-rank newline, insertion order",
        dataset="cl100k_base", datasetSource="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        unicodeClassesSha256=sha(unicode_data),
        unicodeClassesSource={"regexVersion": versions["regex"], "unicodedataVersion": unicodedata.unidata_version,
            "license": "Unicode-3.0", "licenseUrl": "https://www.unicode.org/license.txt",
            "generatorLicenseFile": "REGEX-LICENSE.txt", "unicodeLicenseFile": "UNICODE-LICENSE.txt"},
        attribution={"project": "tiktoken", "copyright": "Copyright (c) 2022 OpenAI, Shantanu Jain", "license": "MIT",
                     "licenseUrl": "https://github.com/openai/tiktoken/blob/main/LICENSE"})
    # Validate/build everything before publication, atomically replace each output,
    # and publish provenance last so an interrupted run cannot claim incomplete data.
    atomic_write(RESOURCES / "cl100k_base.bin.gz", compressed)
    atomic_write(RESOURCES / "unicode-classes.json", unicode_data)
    atomic_write(OUT / "fixtures.json", (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    atomic_write(RESOURCES / "cl100k_base.provenance.json", (json.dumps(provenance, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({key: len(result[key]) for key in ("cases", "decodeCases", "gpt4Cases", "splitCases", "trainErrorCases")}, indent=2))
    print(f"GPT4 merges: {len(gpt.merges)}; packed bytes: {len(packed)}; gzip bytes: {len(compressed)}")


if __name__ == "__main__":
    main()
