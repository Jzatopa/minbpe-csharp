// Unicode-scalar adaptation of minbpe's GPT-style patterns, checked against Python regex.
// Maintained through agentic AI under Jzatopa's direction; MIT license.
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace MinBpe;

public static class TextSplitter
{
    public const string Gpt2Pattern = @"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    public const string Gpt4Pattern = @"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
    internal static void Validate(string pattern)
    {
        if (pattern.Length == 0) throw new ArgumentException("A regex tokenizer requires a nonempty pattern.", nameof(pattern));
        if (pattern == Gpt2Pattern || pattern == Gpt4Pattern) return;
        var regex = new Regex(pattern, RegexOptions.CultureInvariant, TimeSpan.FromSeconds(2));
        if (regex.GetGroupNumbers().Length != 1) throw new ArgumentException("Custom patterns with capturing groups are unsupported; use non-capturing groups.");
    }
    private static bool Letter(Rune r) => UnicodeClasses.Contains("letter", r.Value);
    private static bool Number(Rune r) => UnicodeClasses.Contains("number", r.Value);
    public static IReadOnlyList<string> Split(string text, string? pattern = null)
    {
        _ = Tokenizer.TextBytes(text);
        pattern ??= Gpt4Pattern;
        if (pattern != Gpt2Pattern && pattern != Gpt4Pattern)
        {
            Validate(pattern);
            var matches = Regex.Matches(text, pattern, RegexOptions.CultureInvariant, TimeSpan.FromSeconds(2));
            bool SplitsPair(int offset) => offset > 0 && offset < text.Length && char.IsHighSurrogate(text[offset - 1]) && char.IsLowSurrogate(text[offset]);
            foreach (Match match in matches)
                if (SplitsPair(match.Index) || SplitsPair(match.Index + match.Length))
                    throw new ArgumentException("Custom regex split a UTF-16 surrogate pair. Use a scalar-safe pattern or a built-in GPT pattern.", nameof(pattern));
            return matches.Select(m => m.Value).ToArray();
        }
        bool gpt4 = pattern == Gpt4Pattern;
        var runes = text.EnumerateRunes().ToArray();
        int n = runes.Length;
        var offsets = new int[n + 1];
        for (int k = 0; k < n; k++) offsets[k + 1] = offsets[k] + runes[k].Utf16SequenceLength;
        var chunks = new List<string>();
        bool White(int k) => k < n && UnicodeClasses.Contains("whitespace", runes[k].Value);
        bool L(int k) => k < n && Letter(runes[k]);
        bool N(int k) => k < n && Number(runes[k]);
        bool Punct(int k) => k < n && !White(k) && !L(k) && !N(k);
        bool Newline(int k) => k < n && runes[k].Value is 10 or 13;
        int i = 0;
        while (i < n)
        {
            int end = i;
            // Ordered alternatives and possessive quantifiers match the Python patterns.
            if (runes[i].Value == '\'')
            {
                foreach (var suffix in new[] {"s", "d", "m", "t", "ll", "ve", "re"})
                {
                    if (i + 1 + suffix.Length > n) continue;
                    bool matches = true;
                    for (int k = 0; k < suffix.Length; k++)
                    {
                        int actual = gpt4 ? Rune.ToUpperInvariant(runes[i + 1 + k]).Value : runes[i + 1 + k].Value;
                        int expected = gpt4 ? char.ToUpperInvariant(suffix[k]) : suffix[k];
                        if (actual != expected) { matches = false; break; }
                    }
                    if (matches) { end = i + 1 + suffix.Length; break; }
                }
            }
            if (end == i)
            {
                int start = i;
                if (gpt4 ? !L(i) && !N(i) && !Newline(i) : runes[i].Value == ' ') start++;
                if (L(start)) { end = start + 1; while (L(end)) end++; }
            }
            if (end == i)
            {
                int start = !gpt4 && runes[i].Value == ' ' ? i + 1 : i;
                if (N(start)) { end = start + 1; while (N(end) && (!gpt4 || end - start < 3)) end++; }
            }
            if (end == i)
            {
                int start = runes[i].Value == ' ' ? i + 1 : i;
                if (Punct(start))
                {
                    end = start + 1; while (Punct(end)) end++;
                    if (gpt4) while (Newline(end)) end++;
                }
            }
            if (end == i && gpt4)
            {
                int k = i, lastNewline = -1;
                while (White(k)) { if (Newline(k)) lastNewline = k; k++; }
                if (lastNewline >= 0) end = lastNewline + 1;
            }
            if (end == i && White(i))
            {
                end = i + 1; while (White(end)) end++;
                // \s+(?!\S) backtracks one scalar if followed by non-whitespace.
                if (end < n && end - i > 1) end--;
            }
            if (end == i) throw new InvalidOperationException("Built-in split pattern failed to consume a scalar.");
            chunks.Add(text[offsets[i]..offsets[end]]); i = end;
        }
        return chunks;
    }
}
