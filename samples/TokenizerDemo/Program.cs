using MinBpe;
using System.Text;

bool pretrained = args.FirstOrDefault() == "--gpt4";
string text = string.Join(" ", pretrained ? args.Skip(1) : args);
if (text.Length == 0) text = "Tokenizers turn text into reusable byte pairs. Hello, 世界! 👋";
Tokenizer tokenizer;
if (pretrained) tokenizer = new GPT4Tokenizer();
else
{
    var trained = new RegexTokenizer();
    string corpus = string.Concat(Enumerable.Repeat("Tokenizers turn text into reusable byte pairs. Training a small vocabulary explains how text becomes tokens. Hello, 世界! 👋\n", 8));
    trained.Train(corpus, 280);
    tokenizer = trained;
    Console.WriteLine("Trained vocabulary: 280 tokens on a small demo corpus (not a production tokenizer).");
    foreach (var merge in tokenizer.Merges.Take(5))
        Console.WriteLine($"merge ({merge.Left}, {merge.Right}) -> {merge.Rank}");
}
int[] ids = tokenizer.Encode(text);
Console.WriteLine($"\nInput: {text}\nUTF-8 bytes: {Encoding.UTF8.GetByteCount(text)}; tokens: {ids.Length}");
foreach (int id in ids)
    Console.WriteLine($"{id,6}  {Tokenizer.RenderToken(tokenizer.GetTokenBytes(id))}");
Console.WriteLine($"Round-trip: {tokenizer.Decode(ids) == text}");
