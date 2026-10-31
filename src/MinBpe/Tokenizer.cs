// C# translation of karpathy/minbpe; original MIT license retained in LICENSE.
// This fork is maintained through agentic AI under Jzatopa's direction.
// Changes are checked against Python-derived fixtures and GitHub CI.
using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;

namespace MinBpe;

public readonly record struct MergeRule(int Left, int Right, int Rank);

/// <summary>Base for trainable byte-pair tokenizers. Training/loading mutate the instance.</summary>
public abstract class Tokenizer
{
    internal static readonly UTF8Encoding StrictUtf8 = new(false, true);
    protected Dictionary<(int, int), int> MergeRanks = [];
    protected List<MergeRule> Rules = [];
    protected Dictionary<int, byte[]> Vocab = ByteVocabulary();
    protected List<KeyValuePair<string, int>> Specials = [];
    public string Pattern { get; protected set; } = "";
    public IReadOnlyList<MergeRule> Merges => Rules.AsReadOnly();
    public IReadOnlyDictionary<string, int> SpecialTokens => new ReadOnlyDictionary<string, int>(Specials.ToDictionary());
    public int VocabularySize => 256 + Rules.Count;
    private protected Tokenizer() { }
    public abstract void Train(string text, int vocabularySize, bool verbose = false);
    public abstract int[] Encode(string text);

    internal static byte[] TextBytes(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return StrictUtf8.GetBytes(text);
    }
    protected static Dictionary<int, byte[]> ByteVocabulary() => Enumerable.Range(0, 256).ToDictionary(i => i, i => new[] {(byte)i});
    protected virtual byte[] TokenBytes(int id) => Vocab.TryGetValue(id, out var value) ? value : throw new ArgumentException($"Unknown token ID: {id}", nameof(id));
    public byte[] GetTokenBytes(int id) => (byte[])TokenBytes(id).Clone();
    public virtual string Decode(IEnumerable<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        using var bytes = new MemoryStream();
        foreach (int id in ids) bytes.Write(TokenBytes(id));
        return Encoding.UTF8.GetString(bytes.ToArray());
    }
    protected static int[] Merge(IReadOnlyList<int> ids, (int, int) pair, int rank)
    {
        var result = new List<int>(ids.Count);
        for (int i = 0; i < ids.Count; i++)
        {
            if (i + 1 < ids.Count && ids[i] == pair.Item1 && ids[i + 1] == pair.Item2)
            { result.Add(rank); i++; }
            else result.Add(ids[i]);
        }
        return result.ToArray();
    }
    protected void TrainChunks(List<int[]> chunks, int vocabularySize, bool verbose)
    {
        if (vocabularySize < 256) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (Specials.Any(p => p.Value < vocabularySize)) throw new ArgumentException("Special token ID overlaps the requested vocabulary.");
        var rules = new List<MergeRule>();
        var ranks = new Dictionary<(int, int), int>();
        var vocab = ByteVocabulary();
        for (int rank = 256; rank < vocabularySize; rank++)
        {
            var counts = new Dictionary<(int, int), int>();
            var order = new List<(int, int)>();
            foreach (var chunk in chunks)
                for (int i = 0; i + 1 < chunk.Length; i++)
                {
                    var pair = (chunk[i], chunk[i + 1]);
                    if (!counts.ContainsKey(pair)) { counts[pair] = 0; order.Add(pair); }
                    counts[pair]++;
                }
            if (order.Count == 0) throw new InvalidOperationException("Training text has no remaining pairs; request a smaller vocabulary.");
            var best = order[0];
            foreach (var pair in order) if (counts[pair] > counts[best]) best = pair;
            chunks = chunks.Select(chunk => Merge(chunk, best, rank)).ToList();
            ranks[best] = rank;
            rules.Add(new(best.Item1, best.Item2, rank));
            vocab[rank] = [..vocab[best.Item1], ..vocab[best.Item2]];
            if (verbose) Console.WriteLine($"merge {rank - 255}/{vocabularySize - 256}: {best} -> {rank} ({counts[best]} occurrences)");
        }
        Rules = rules; MergeRanks = ranks; Vocab = vocab;
    }
    protected virtual int[] EncodeChunk(byte[] bytes)
    {
        int[] ids = bytes.Select(b => (int)b).ToArray();
        while (ids.Length > 1)
        {
            (int, int) best = default;
            int lowest = int.MaxValue;
            for (int i = 0; i + 1 < ids.Length; i++)
                if (MergeRanks.TryGetValue((ids[i], ids[i + 1]), out int rank) && rank < lowest)
                { best = (ids[i], ids[i + 1]); lowest = rank; }
            if (lowest == int.MaxValue) break;
            ids = Merge(ids, best, lowest);
        }
        return ids;
    }
    public virtual void Save(string filePrefix)
    {
        ArgumentException.ThrowIfNullOrEmpty(filePrefix);
        if (Pattern != TrimPythonWhitespace(Pattern) || Pattern.Contains('\n') || Pattern.Contains('\r')) throw new InvalidOperationException("Model patterns must fit on one line without leading or trailing whitespace.");
        var model = new StringBuilder("minbpe v1\n").Append(Pattern).Append('\n').Append(Specials.Count).Append('\n');
        foreach (var item in Specials) model.Append(item.Key).Append(' ').Append(item.Value).Append('\n');
        foreach (var rule in Rules) model.Append(rule.Left).Append(' ').Append(rule.Right).Append('\n');
        File.WriteAllText(filePrefix + ".model", model.ToString(), StrictUtf8);
        SaveVocab(filePrefix + ".vocab");
    }
    public virtual void SaveVocab(string path) => WriteVocab(path, Vocab);
    protected void WriteVocab(string path, Dictionary<int, byte[]> vocab)
    {
        var inverse = Rules.ToDictionary(r => r.Rank);
        var text = new StringBuilder();
        foreach (var (id, bytes) in vocab)
        {
            if (inverse.TryGetValue(id, out var rule))
                text.Append('[').Append(RenderToken(vocab[rule.Left])).Append("][").Append(RenderToken(vocab[rule.Right])).Append("] -> ");
            text.Append('[').Append(RenderToken(bytes)).Append("] ").Append(id).Append('\n');
        }
        File.WriteAllText(path, text.ToString(), StrictUtf8);
    }
    public static string RenderToken(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        var text = new StringBuilder();
        foreach (var rune in Encoding.UTF8.GetString(bytes).EnumerateRunes())
        {
            if (UnicodeClasses.Contains("control", rune.Value))
                text.Append("\\u").Append(rune.Value.ToString("x4", CultureInfo.InvariantCulture));
            else text.Append(rune.ToString());
        }
        return text.ToString();
    }
    public virtual void Load(string modelFile)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelFile);
        if (!modelFile.EndsWith(".model", StringComparison.Ordinal)) throw new ArgumentException("Expected a .model file.", nameof(modelFile));
        var raw = File.ReadAllBytes(modelFile);
        if (raw.AsSpan().StartsWith(new byte[] { 0xef, 0xbb, 0xbf })) throw new FormatException("Model files must not contain a UTF-8 BOM.");
        var lines = File.ReadAllLines(modelFile, StrictUtf8);
        if (lines.Length < 3 || TrimPythonWhitespace(lines[0]) != "minbpe v1") throw new FormatException("Unsupported model header.");
        string pattern = lines[1];
        if (pattern != TrimPythonWhitespace(pattern)) throw new FormatException("Model pattern has ambiguous leading or trailing whitespace.");
        ValidatePattern(pattern);
        if (!int.TryParse(TrimPythonWhitespace(lines[2]), NumberStyles.Integer, CultureInfo.InvariantCulture, out int specialCount) || specialCount < 0 || specialCount > lines.Length - 3)
            throw new FormatException("Invalid special-token count.");
        var specials = new List<KeyValuePair<string, int>>();
        for (int i = 0; i < specialCount; i++)
        {
            var parts = lines[3 + i].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 2 || !int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int id)) throw new FormatException("Invalid special-token entry.");
            specials.Add(new(parts[0], id));
        }
        var rules = new List<MergeRule>();
        var ranks = new Dictionary<(int, int), int>();
        var vocab = ByteVocabulary();
        for (int i = 3 + specialCount; i < lines.Length; i++)
        {
            var parts = lines[i].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 2 || !int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int left) || !int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int right) || !vocab.ContainsKey(left) || !vocab.ContainsKey(right) || ranks.ContainsKey((left, right)))
                throw new FormatException("Invalid or out-of-order merge entry.");
            int rank = 256 + rules.Count;
            ranks[(left, right)] = rank; rules.Add(new(left, right, rank)); vocab[rank] = [..vocab[left], ..vocab[right]];
        }
        ValidateSpecials(specials, vocab.Count);
        foreach (var item in specials) vocab[item.Value] = TextBytes(item.Key);
        Pattern = pattern; Specials = specials; Rules = rules; MergeRanks = ranks; Vocab = vocab;
    }
    protected virtual void ValidatePattern(string pattern) { }
    private static bool PythonWhitespace(char c) => UnicodeClasses.Contains("whitespace", c) || c is >= '\u001c' and <= '\u001f';
    private static string TrimPythonWhitespace(string value)
    {
        int start = 0, end = value.Length;
        while (start < end && PythonWhitespace(value[start])) start++;
        while (end > start && PythonWhitespace(value[end - 1])) end--;
        return value[start..end];
    }
    protected static void ValidateSpecials(List<KeyValuePair<string, int>> specials, int vocabularySize)
    {
        var names = new HashSet<string>(StringComparer.Ordinal); var ids = new HashSet<int>();
        foreach (var (token, id) in specials)
            if (string.IsNullOrEmpty(token) || token.Any(PythonWhitespace) || !names.Add(token) || id < vocabularySize || !ids.Add(id))
                throw new ArgumentException("Special tokens must have unique nonempty whitespace-free strings and unique IDs outside the normal vocabulary.");
    }
}
