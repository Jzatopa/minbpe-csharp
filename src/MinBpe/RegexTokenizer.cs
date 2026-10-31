// MIT-licensed C# minbpe port; maintained through agentic AI under Jzatopa's direction.
namespace MinBpe;

public enum SpecialTokenMode { NoneRaise, None, All }

public class RegexTokenizer : Tokenizer
{
    public RegexTokenizer(string? pattern = null)
    {
        Pattern = pattern ?? TextSplitter.Gpt4Pattern;
        ValidatePattern(Pattern);
    }
    protected override void ValidatePattern(string pattern) => TextSplitter.Validate(pattern);
    public override void Train(string text, int vocabularySize, bool verbose = false) =>
        TrainChunks(TextSplitter.Split(text, Pattern).Select(chunk => TextBytes(chunk).Select(b => (int)b).ToArray()).ToList(), vocabularySize, verbose);
    public void RegisterSpecialTokens(IEnumerable<KeyValuePair<string, int>> specials)
    {
        ArgumentNullException.ThrowIfNull(specials);
        var items = specials.ToList();
        ValidateSpecials(items, 256 + Rules.Count);
        foreach (var item in items) _ = TextBytes(item.Key);
        // Loaded models contain special bytes in Vocab; discard stale mappings on re-registration.
        int normalSize = 256 + Rules.Count;
        if (Vocab.Keys.Any(id => id >= normalSize))
            Vocab = Vocab.Where(item => item.Key < normalSize).ToDictionary();
        Specials = items;
    }
    protected override byte[] TokenBytes(int id)
    {
        if (Vocab.TryGetValue(id, out var bytes)) return bytes;
        foreach (var item in Specials) if (item.Value == id) return TextBytes(item.Key);
        throw new ArgumentException($"Unknown token ID: {id}", nameof(id));
    }
    public int[] EncodeOrdinary(string text)
    {
        var result = new List<int>();
        foreach (string chunk in TextSplitter.Split(text, Pattern)) result.AddRange(EncodeChunk(TextBytes(chunk)));
        return result.ToArray();
    }
    public override int[] Encode(string text) => Encode(text, SpecialTokenMode.NoneRaise);
    public int[] Encode(string text, SpecialTokenMode mode)
    {
        _ = TextBytes(text);
        if (!Enum.IsDefined(mode)) throw new ArgumentOutOfRangeException(nameof(mode));
        if (mode == SpecialTokenMode.NoneRaise && Specials.Any(item => text.Contains(item.Key, StringComparison.Ordinal)))
            throw new ArgumentException("A registered special token is present but not explicitly allowed.", nameof(text));
        return mode == SpecialTokenMode.All ? EncodeAllowed(text, Specials) : EncodeOrdinary(text);
    }
    public int[] Encode(string text, IReadOnlySet<string> allowedSpecial)
    {
        ArgumentNullException.ThrowIfNull(allowedSpecial);
        _ = TextBytes(text);
        var ordinalAllowed = new HashSet<string>(allowedSpecial, StringComparer.Ordinal);
        return EncodeAllowed(text, Specials.Where(item => ordinalAllowed.Contains(item.Key)).ToList());
    }
    private int[] EncodeAllowed(string text, List<KeyValuePair<string, int>> active)
    {
        if (active.Count == 0) return EncodeOrdinary(text);
        var result = new List<int>();
        int start = 0;
        while (start < text.Length)
        {
            int next = -1; KeyValuePair<string, int> selected = default;
            foreach (var item in active)
            {
                int found = text.IndexOf(item.Key, start, StringComparison.Ordinal);
                if (found >= 0 && (next < 0 || found < next)) { next = found; selected = item; }
            }
            if (next < 0) { result.AddRange(EncodeOrdinary(text[start..])); break; }
            result.AddRange(EncodeOrdinary(text[start..next])); result.Add(selected.Value);
            start = next + selected.Key.Length;
        }
        return result.ToArray();
    }
}
