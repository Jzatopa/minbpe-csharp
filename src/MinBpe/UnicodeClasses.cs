// Pinned scalar classifications exported from Python regex/unicodedata.
// Maintained through agentic AI; see packaged provenance and Unicode attribution.
using System.Text.Json;

namespace MinBpe;

internal static class UnicodeClasses
{
    private static readonly Dictionary<string, int[][]> Classes = Read();
    private static Dictionary<string, int[][]> Read()
    {
        using var resource = typeof(UnicodeClasses).Assembly.GetManifestResourceStream("MinBpe.Resources.unicode-classes.json") ?? throw new InvalidOperationException("Missing Unicode class resource.");
        using var json = JsonDocument.Parse(resource);
        var classes = new Dictionary<string, int[][]>();
        foreach (var entry in json.RootElement.GetProperty("classes").EnumerateObject())
        {
            var ranges = entry.Value.Deserialize<int[][]>()!;
            int previousEnd = -1;
            foreach (var range in ranges)
            {
                if (range.Length != 2 || range[0] <= previousEnd || range[0] > range[1] || range[1] > 0x10ffff)
                    throw new FormatException("Invalid Unicode class range.");
                previousEnd = range[1];
            }
            classes[entry.Name] = ranges;
        }
        return classes;
    }
    public static bool Contains(string category, int scalar)
    {
        var ranges = Classes[category];
        int lo = 0, hi = ranges.Length - 1;
        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            if (scalar < ranges[mid][0]) hi = mid - 1;
            else if (scalar > ranges[mid][1]) lo = mid + 1;
            else return true;
        }
        return false;
    }
}
