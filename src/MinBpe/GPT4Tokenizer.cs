// C# adaptation of minbpe's GPT4Tokenizer with embedded, pinned cl100k_base merges.
// MIT attribution and data provenance are packaged; no runtime downloads or Python.
// Maintained through agentic AI under Jzatopa's direction.
using System.IO.Compression;
using System.Text;

namespace MinBpe;

public sealed class GPT4Tokenizer : RegexTokenizer
{
    private sealed record Data(List<MergeRule> Rules, Dictionary<(int, int), int> Ranks, Dictionary<int, byte[]> Vocab, byte[] Shuffle, byte[] Inverse);
    private static readonly Lazy<Data> Shared = new(ReadData);
    private readonly byte[] shuffle;
    private readonly byte[] inverse;
    public GPT4Tokenizer() : base(TextSplitter.Gpt4Pattern)
    {
        var data = Shared.Value;
        Rules = data.Rules; MergeRanks = data.Ranks; Vocab = data.Vocab;
        shuffle = data.Shuffle; inverse = data.Inverse;
        RegisterSpecialTokens(new Dictionary<string, int> {
            ["<|endoftext|>"] = 100257, ["<|fim_prefix|>"] = 100258,
            ["<|fim_middle|>"] = 100259, ["<|fim_suffix|>"] = 100260, ["<|endofprompt|>"] = 100276
        });
    }
    private static Data ReadData()
    {
        using var resource = typeof(GPT4Tokenizer).Assembly.GetManifestResourceStream("MinBpe.Resources.cl100k_base.bin.gz") ?? throw new InvalidOperationException("Missing pretrained resource.");
        using var gzip = new GZipStream(resource, CompressionMode.Decompress);
        using var reader = new BinaryReader(gzip, Encoding.UTF8);
        if (Encoding.ASCII.GetString(reader.ReadBytes(8)) != "MINBPE01") throw new FormatException("Bad pretrained resource header.");
        int count = reader.ReadInt32();
        if (count != 100000) throw new FormatException("Unexpected pretrained merge count.");
        var shuffle = reader.ReadBytes(256);
        if (shuffle.Length != 256 || shuffle.Distinct().Count() != 256) throw new FormatException("Invalid byte permutation.");
        var inverse = new byte[256];
        for (int i = 0; i < 256; i++) inverse[shuffle[i]] = (byte)i;
        var rules = new List<MergeRule>(); var ranks = new Dictionary<(int, int), int>(); var vocab = ByteVocabulary();
        for (int i = 0; i < count; i++)
        {
            int left = reader.ReadInt32(), right = reader.ReadInt32(), rank = reader.ReadInt32();
            if (rank != i + 256 || !vocab.ContainsKey(left) || !vocab.ContainsKey(right) || !ranks.TryAdd((left, right), rank))
                throw new FormatException("Invalid pretrained merge rule.");
            rules.Add(new(left, right, rank)); vocab[rank] = [..vocab[left], ..vocab[right]];
        }
        if (gzip.ReadByte() != -1) throw new FormatException("Trailing pretrained resource bytes.");
        return new(rules, ranks, vocab, shuffle, inverse);
    }
    protected override int[] EncodeChunk(byte[] bytes) => base.EncodeChunk(bytes.Select(b => shuffle[b]).ToArray());
    protected override byte[] TokenBytes(int id) => Vocab.TryGetValue(id, out var bytes)
        ? bytes.Select(b => inverse[b]).ToArray() : throw new ArgumentException($"Unknown pretrained token ID: {id}", nameof(id));
    public override void Train(string text, int vocabularySize, bool verbose = false) => throw new NotSupportedException("Pretrained tokenizer cannot be trained.");
    public override void Save(string filePrefix) => throw new NotSupportedException("Pretrained tokenizer cannot be saved as a minbpe v1 model.");
    public override void Load(string modelFile) => throw new NotSupportedException("Pretrained tokenizer cannot load a minbpe v1 model.");
    public override void SaveVocab(string path)
    {
        var actual = Vocab.ToDictionary(item => item.Key, item => item.Value.Select(b => inverse[b]).ToArray());
        WriteVocab(path, actual);
    }
}
