# MinBpe.CSharp

A managed **C# / .NET 10** port of [Andrej Karpathy's minbpe](https://github.com/karpathy/minbpe): train byte-pair tokenizers, inspect merges, encode/decode text, exchange model files, and use the pinned `cl100k_base` pretrained tokenizer **offline**.

Independent fork, not an official upstream release. Original Python source/history and MIT attribution are retained. [Original README](docs/UPSTREAM_README.md) · [Compatibility contract](docs/COMPATIBILITY.md) · [License](LICENSE)

## Maintenance

This fork is maintained through agentic AI under Jzatopa's direction. AI agents contribute implementation, review, tests and documentation; changes are checked against the original Python implementation and GitHub CI. Completed conversions are enrolled in Crossy's monthly upstream-maintenance register. This describes the fork, not the upstream project.

## Why C# instead of Python?

- **Embed training and tokenization in .NET applications.** A desktop tokenizer explorer, ASP.NET service, or local document pipeline can use the same C# types, debugger and deployment process as the rest of the application—without a Python subprocess or service.
- **No runtime NuGet dependencies, Python, or network calls.** The pretrained merge data and pinned Unicode classifications are bundled, with provenance and licenses. Python/regex/tiktoken are used only to generate development reference fixtures.
- **Train your own small tokenizer.** This is more than token counting: `BasicTokenizer` and `RegexTokenizer` expose learned merge rules and portable minbpe model persistence, useful for education and experimentation.
- **A verifiable cross-language port.** Compatibility checks compare exact merge order, token IDs and saved files against Python-derived fixtures, including Unicode and special-token policies.

**Python remains preferable** if your workflow already lives in Python or you want to follow the original tutorial directly. This does not improve model quality, make pretrained GPT models trainable, or imply a measured speed advantage. For a production-focused .NET tokenizer, also evaluate [Microsoft.ML.Tokenizers](https://github.com/dotnet/machinelearning/tree/main/src/Microsoft.ML.Tokenizers). This project's distinction is minbpe-style training, model-file compatibility and a small inspectable implementation.

## Quick start

Requires .NET 10 SDK. The package is available as a GitHub release asset/local package, **not published to NuGet.org**.

```bash
git clone https://github.com/Jzatopa/minbpe-csharp.git
cd minbpe-csharp
dotnet run --project samples/TokenizerDemo -c Release
dotnet run --project samples/TokenizerDemo -c Release -- --gpt4 "Hello, 世界! 👋"
```

The demo shows training merges or pretrained tokens, token bytes and round-trip decoding. Its small training corpus is illustrative, not a production tokenizer.

### Train, encode and save

```csharp
using MinBpe;

var tokenizer = new BasicTokenizer();
tokenizer.Train("aaabdaaabac", vocabularySize: 259);
int[] ids = tokenizer.Encode("aaabdaaabac");
// [258, 100, 258, 97, 99]
string original = tokenizer.Decode(ids);
foreach (var merge in tokenizer.Merges)
    Console.WriteLine($"({merge.Left}, {merge.Right}) -> {merge.Rank}");

tokenizer.Save("example"); // example.model and display-only example.vocab
var restored = new BasicTokenizer();
restored.Load("example.model");
```

`RegexTokenizer` trains within GPT-style text chunks. It defaults to the source's GPT4 pattern; pass `TextSplitter.Gpt2Pattern` for GPT2 splitting or a supported non-capturing .NET pattern for custom splitting.

### Special tokens

```csharp
var tokenizer = new RegexTokenizer();
tokenizer.RegisterSpecialTokens(new Dictionary<string, int> { ["<|doc|>"] = 1000 });
int[] ids = tokenizer.Encode("<|doc|>Hello", SpecialTokenMode.All);
string text = tokenizer.Decode(ids); // "<|doc|>Hello"

// Default Encode rejects any registered special token in the text.
// None treats specials as ordinary text; an explicit set allows selected specials.
int[] ordinary = tokenizer.Encode("<|doc|>Hello", SpecialTokenMode.None);
int[] selected = tokenizer.Encode("<|doc|>Hello", new HashSet<string> { "<|doc|>" });
```

### Pretrained encoding

```csharp
var tokenizer = new GPT4Tokenizer();
int[] ids = tokenizer.Encode("hello world"); // [15339, 1917]
string text = tokenizer.Decode(ids);
```

The name follows upstream. It specifically represents **`cl100k_base`**, not every GPT model or current model-specific tokenizer. Like upstream, pretrained training and minbpe model save/load are unsupported; `SaveVocab(path)` is supported. Also like upstream, **decoding pretrained special-token IDs raises**; see the compatibility contract before using `SpecialTokenMode.All` with this class.

## Verify and package

```bash
dotnet build src/MinBpe -c Release
dotnet run --project tests/MinBpe.Compatibility -c Release
dotnet pack src/MinBpe -c Release -o artifacts
```

The test suite is an executable harness: use `dotnet run`, **not `dotnet test`**. It exits nonzero on failure and requires no Python or network access. It checks 134 trained models, 2,706 encode comparisons (trained, loaded and pretrained), 137 decode cases, 111 split cases, 16 training failures, and C# API/persistence invariants. Expected IDs, merge order, and model/vocabulary files are generated by the original Python code. Pretrained ordinary-token fixtures are additionally compared to tiktoken during generation.

To reproduce the oracle and original Python tests, use Python **3.12.3**, matching the recorded environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-oracle.txt
.venv/bin/python scripts/generate-fixtures.py
.venv/bin/python -m pytest -q tests/test_tokenizer.py
```

Oracle regeneration downloads tiktoken's public encoding data on first use. The resulting library does not. [Oracle schema and generation details](scripts/README.md).

GitHub Actions regenerates oracle/resources, rejects unexpected diffs, runs upstream and C# tests, executes both demo modes, and builds a package artifact.

## Attribution and scope

Pinned minbpe revision: [`1acefe8`](https://github.com/karpathy/minbpe/commit/1acefe89412b20245db5a22d2a02001e547dc602). The fork retains the original MIT license. Embedded tiktoken-derived data, Unicode data and regex-derived classification tables retain their respective notices/licenses; see [NOTICE](NOTICE) and `src/MinBpe/Resources/`.

Model loading fixes the original implementation's stale regex/inverse-special-map behavior. Custom regex syntax and validation differ deliberately; consult [the compatibility contract](docs/COMPATIBILITY.md). No universal compatibility with arbitrary Python `regex` expressions is claimed.
