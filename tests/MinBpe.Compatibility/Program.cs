using MinBpe;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text;
using System.IO.Compression;

int assertions = 0, encodeChecks = 0;
void Check(bool condition, string label)
{
    assertions++;
    if (!condition) throw new Exception("FAIL: " + label);
}
void Throws<T>(Action action, string label) where T : Exception
{
    bool threw = false;
    try { action(); }
    catch (T) { threw = true; }
    Check(threw, label + " should throw");
}
string? Pattern(JsonElement test) => test.GetProperty("pattern").ValueKind == JsonValueKind.Null ? null : test.GetProperty("pattern").GetString();
Tokenizer Create(JsonElement test) => test.GetProperty("kind").GetString() == "basic" ? new BasicTokenizer() : new RegexTokenizer(Pattern(test));
int[] Encode(Tokenizer tokenizer, JsonElement query)
{
    string text = query.GetProperty("text").GetString()!;
    if (tokenizer is not RegexTokenizer regex) return tokenizer.Encode(text);
    return query.GetProperty("mode").GetString() switch
    {
        "all" => regex.Encode(text, SpecialTokenMode.All),
        "none" => regex.Encode(text, SpecialTokenMode.None),
        "none_raise" => regex.Encode(text),
        "set" => regex.Encode(text, query.GetProperty("allowed").Deserialize<string[]>()!.ToHashSet(StringComparer.Ordinal)),
        _ => throw new Exception("Unknown fixture mode")
    };
}
void Query(Tokenizer tokenizer, JsonElement query, string label)
{
    encodeChecks++;
    if (query.TryGetProperty("error", out var error) && error.GetBoolean()) Throws<ArgumentException>(() => Encode(tokenizer, query), label);
    else
    {
        int[] expected = query.GetProperty("ids").Deserialize<int[]>()!;
        int[] actual = Encode(tokenizer, query);
        Check(actual.SequenceEqual(expected), label + $" IDs: expected [{string.Join(',', expected)}], got [{string.Join(',', actual)}]");
    }
}

using var fixtures = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures.json")));
var root = fixtures.RootElement;
const string sourceCommit = "1acefe89412b20245db5a22d2a02001e547dc602";
Check(root.GetProperty("sourceCommit").GetString() == sourceCommit, "pinned source commit");
Check(root.GetProperty("sourceHashes").EnumerateObject().Count() == 5, "all five reference source hashes present");
Check(root.GetProperty("cases").EnumerateArray().Sum(test => test.GetProperty("queries").GetArrayLength()) == 1268, "all reference queries present");
foreach (var file in root.GetProperty("sourceHashes").EnumerateObject())
{
    var actual = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "oracle-source", file.Name)))).ToLowerInvariant();
    Check(actual == file.Value.GetString(), "reference source hash " + file.Name);
}
foreach (var (name, count) in new[] {("cases",134),("decodeCases",137),("gpt4Cases",170),("splitCases",113),("trainErrorCases",16)})
    Check(root.GetProperty(name).GetArrayLength() == count, "fixture completeness " + name);
using var provenance = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "cl100k_base.provenance.json")));
var packed = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "cl100k_base.bin.gz"));
Check(Convert.ToHexString(SHA256.HashData(packed)).Equals(provenance.RootElement.GetProperty("compressedSha256").GetString(), StringComparison.OrdinalIgnoreCase), "pretrained resource hash");
Check(provenance.RootElement.GetProperty("sourceCommit").GetString() == sourceCommit, "pretrained source pin");
Check(Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "unicode-classes.json"))))
    .Equals(provenance.RootElement.GetProperty("unicodeClassesSha256").GetString(), StringComparison.OrdinalIgnoreCase), "pinned Unicode resource hash");
using (var gzip = new GZipStream(new MemoryStream(packed), CompressionMode.Decompress))
using (var bytes = new MemoryStream())
{
    gzip.CopyTo(bytes);
    Check(Convert.ToHexString(SHA256.HashData(bytes.ToArray())).Equals(provenance.RootElement.GetProperty("uncompressedSha256").GetString(), StringComparison.OrdinalIgnoreCase), "uncompressed pretrained hash");
}

string temp = Path.Combine(Path.GetTempPath(), "minbpe-csharp-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(temp);
try
{
    foreach (var test in root.GetProperty("cases").EnumerateArray())
    {
        string name = test.GetProperty("name").GetString()!;
        Tokenizer tokenizer = Create(test);
        tokenizer.Train(test.GetProperty("trainText").GetString()!, test.GetProperty("vocabSize").GetInt32());
        Check(tokenizer.VocabularySize == test.GetProperty("vocabSize").GetInt32(), name + " normal vocabulary size");
        var expectedMerges = test.GetProperty("merges").EnumerateArray().Select(row => new MergeRule(row[0].GetInt32(), row[1].GetInt32(), row[2].GetInt32()));
        Check(tokenizer.Merges.SequenceEqual(expectedMerges), name + " training merge order");
        var specials = test.GetProperty("specials").Deserialize<Dictionary<string, int>>()!;
        if (tokenizer is RegexTokenizer regex) regex.RegisterSpecialTokens(specials);
        string prefix = Path.Combine(temp, "trained");
        tokenizer.Save(prefix);
        Check(File.ReadAllText(prefix + ".model") == test.GetProperty("model").GetString(), name + " Python model file");
        Check(File.ReadAllText(prefix + ".vocab") == test.GetProperty("vocab").GetString(), name + " Python vocabulary display file");
        File.WriteAllText(prefix + ".model", test.GetProperty("model").GetString(), new UTF8Encoding(false));
        // Intentionally use the default constructor: Load must restore non-default patterns and specials.
        Tokenizer loaded = test.GetProperty("kind").GetString() == "basic" ? new BasicTokenizer() : new RegexTokenizer();
        loaded.Load(prefix + ".model");
        Check(loaded.VocabularySize == tokenizer.VocabularySize, name + " loaded normal vocabulary size");
        loaded.Save(prefix + "-loaded");
        Check(File.ReadAllText(prefix + "-loaded.model") == test.GetProperty("model").GetString(), name + " load/save model stability");
        foreach (var query in test.GetProperty("queries").EnumerateArray())
        {
            Query(tokenizer, query, name + " trained");
            Query(loaded, query, name + " loaded");
            if (!query.TryGetProperty("error", out _))
            {
                Check(tokenizer.Decode(Encode(tokenizer, query)) == query.GetProperty("text").GetString(), name + " trained roundtrip");
                Check(loaded.Decode(Encode(loaded, query)) == query.GetProperty("text").GetString(), name + " loaded roundtrip");
            }
        }
    }
    int splitCount = 0;
    foreach (var test in root.GetProperty("splitCases").EnumerateArray())
    {
        var actual = TextSplitter.Split(test.GetProperty("text").GetString()!, Pattern(test));
        var expected = test.GetProperty("chunks").Deserialize<string[]>()!;
        Check(actual.SequenceEqual(expected), $"split {splitCount}: expected {JsonSerializer.Serialize(expected)}, got {JsonSerializer.Serialize(actual)}");
        splitCount++;
    }
    var gpt4 = new GPT4Tokenizer();
    foreach (var test in root.GetProperty("gpt4Cases").EnumerateArray()) Query(gpt4, test, "GPT4 " + test.GetProperty("text").GetString());
    foreach (var test in root.GetProperty("decodeCases").EnumerateArray())
    {
        Tokenizer tokenizer = test.GetProperty("kind").GetString() switch
        { "basic" => new BasicTokenizer(), "regex" => new RegexTokenizer(), _ => gpt4 };
        int[] ids = test.GetProperty("ids").Deserialize<int[]>()!;
        if (test.TryGetProperty("error", out var error) && error.GetBoolean()) Throws<ArgumentException>(() => tokenizer.Decode(ids), "decode error");
        else Check(tokenizer.Decode(ids) == test.GetProperty("text").GetString(), "decode " + string.Join(',', ids));
    }
    foreach (var test in root.GetProperty("trainErrorCases").EnumerateArray())
    {
        if (test.GetProperty("vocabSize").GetInt32() < 256)
            Throws<ArgumentOutOfRangeException>(() => Create(test).Train(test.GetProperty("text").GetString()!, test.GetProperty("vocabSize").GetInt32()), "invalid vocabulary size");
        else Throws<InvalidOperationException>(() => Create(test).Train(test.GetProperty("text").GetString()!, test.GetProperty("vocabSize").GetInt32()), "exhausted training");
    }
    // C#-specific validation and persistence invariants, independent of oracle comparisons.
    var basic = new BasicTokenizer();
    Throws<ArgumentException>(() => basic.Encode("\ud800"), "reject lone UTF-16 surrogate");
    Throws<ArgumentException>(() => new RegexTokenizer("(capture)"), "reject custom captures");
    var regexTokenizer = new RegexTokenizer();
    Throws<ArgumentException>(() => regexTokenizer.RegisterSpecialTokens(new Dictionary<string,int> { ["bad token"] = 1000 }), "reject whitespace special");
    Throws<ArgumentException>(() => regexTokenizer.RegisterSpecialTokens(new Dictionary<string,int> { ["<x>"] = 0 }), "reject special ID collision");
    Throws<ArgumentException>(() => regexTokenizer.RegisterSpecialTokens(new Dictionary<string,int> { ["x\u001cy"] = 1000 }), "reject Python split whitespace in special names");
    Throws<NotSupportedException>(() => gpt4.Train("hello", 256), "pretrained training unsupported");
    Throws<NotSupportedException>(() => gpt4.Save(Path.Combine(temp,"gpt4")), "pretrained model saving unsupported");
    Throws<NotSupportedException>(() => gpt4.Load(Path.Combine(temp,"trained.model")), "pretrained loading unsupported");
    gpt4.SaveVocab(Path.Combine(temp,"gpt4.vocab"));
    Check(File.ReadLines(Path.Combine(temp,"gpt4.vocab")).Count() == 100256, "pretrained vocabulary export");
    Check(Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(temp,"gpt4.vocab")))).Equals(
        provenance.RootElement.GetProperty("gpt4VocabSha256").GetString(), StringComparison.OrdinalIgnoreCase), "complete pretrained vocabulary matches Python export");
    var reloaded = new RegexTokenizer();
    reloaded.RegisterSpecialTokens(new Dictionary<string,int> { ["<x>"] = 1000 });
    reloaded.Save(Path.Combine(temp, "special"));
    reloaded.Load(Path.Combine(temp, "special.model"));
    Check(reloaded.Decode(new[] {1000}) == "<x>", "load reconstructs inverse special state");
    reloaded.RegisterSpecialTokens(new Dictionary<string,int> { ["<y>"] = 1000 });
    Check(reloaded.Decode(new[] {1000}) == "<y>", "re-registration replaces stale loaded bytes");
    var caseInsensitive = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "<Y>" };
    Check(reloaded.Encode("<y>", caseInsensitive).SequenceEqual(reloaded.EncodeOrdinary("<y>")), "allow-set membership remains ordinal");
    Throws<InvalidOperationException>(() => new RegexTokenizer(" leading ").Save(Path.Combine(temp, "unsafe")), "reject non-roundtrippable pattern");
    File.WriteAllText(Path.Combine(temp,"bad.model"), "minbpe v1\n\n0\n999999 0\n");
    basic.Train("aaabdaaabac", 259);
    var before = basic.Encode("aaabdaaabac");
    var beforeMerges = basic.Merges.ToArray();
    Throws<FormatException>(() => basic.Load(Path.Combine(temp,"bad.model")), "bad merge model rejected");
    Check(basic.Encode("aaabdaaabac").SequenceEqual(before) && basic.Merges.SequenceEqual(beforeMerges), "failed load preserves trained state");
    foreach (string contents in new[] {
        "wrong header\n\n0\n", "minbpe v1\n\n2\n", "minbpe v1\n\n0\n97 98\n97 98\n",
        "\ufeffminbpe v1\n\n0\n", "minbpe v1\n leading \n0\n" })
    {
        File.WriteAllText(Path.Combine(temp, "bad.model"), contents, new UTF8Encoding(false));
        Throws<FormatException>(() => basic.Load(Path.Combine(temp, "bad.model")), "malformed model rejected");
        Check(basic.Merges.SequenceEqual(beforeMerges), "invalid model retains merge state");
    }
    reloaded.Train("aaabdaaabac", 259);
    var previousPattern = reloaded.Pattern;
    var previousMerges = reloaded.Merges.ToArray();
    var previousSpecials = reloaded.SpecialTokens.ToArray();
    File.WriteAllText(Path.Combine(temp,"bad.model"), "minbpe v1\n(capture)\n0\n");
    Throws<ArgumentException>(() => reloaded.Load(Path.Combine(temp,"bad.model")), "invalid custom pattern rejected on load");
    Check(reloaded.Pattern == previousPattern && reloaded.Merges.SequenceEqual(previousMerges) && reloaded.SpecialTokens.SequenceEqual(previousSpecials), "failed regex load preserves complete state");
    File.WriteAllText(Path.Combine(temp,"bad.model"), "minbpe v1\n\n0\n");
    Throws<ArgumentException>(() => reloaded.Load(Path.Combine(temp,"bad.model")), "regex rejects empty Basic pattern on load");
    File.WriteAllText(Path.Combine(temp,"bad.model"), "minbpe v1\n" + TextSplitter.Gpt4Pattern + "\n1\n<x> 256\n97 98\n");
    Throws<ArgumentException>(() => reloaded.Load(Path.Combine(temp,"bad.model")), "special ID collision rejected on load");
    Throws<EncoderFallbackException>(() => reloaded.Encode("\ud800"), "regex rejects lone surrogate");
    Throws<EncoderFallbackException>(() => reloaded.Train("\ud800",256), "regex training rejects lone surrogate");
    Throws<EncoderFallbackException>(() => TextSplitter.Split("\ud800"), "splitter rejects lone surrogate");
    Throws<ArgumentException>(() => new RegexTokenizer(".").Encode("😀"), "unsafe custom surrogate split rejected clearly");
    reloaded.RegisterSpecialTokens(new Dictionary<string,int> { ["<a>"] = 1000, ["<b>"] = 1001 });
    Check(reloaded.Encode("<a><b>", new HashSet<string>()).SequenceEqual(reloaded.EncodeOrdinary("<a><b>")), "empty allow-set");
    Check(reloaded.Encode("<a><b>", new HashSet<string> {"<unknown>"}).SequenceEqual(reloaded.EncodeOrdinary("<a><b>")), "unknown allow-set names ignored");
    Check(reloaded.Encode("<a><b>", new HashSet<string> {"<a>","<b>"}).SequenceEqual(new[] {1000,1001}), "all-special allow-set");
}
finally { Directory.Delete(temp, recursive:true); }
Console.WriteLine($"PASS: 134 trained models, {encodeChecks} encode comparisons, 137 decode cases, 113 splits, 16 training failures; {assertions} assertions.");
