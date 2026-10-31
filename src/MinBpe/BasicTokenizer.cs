// MIT-licensed C# port; maintained through agentic AI under Jzatopa's direction.
namespace MinBpe;

public sealed class BasicTokenizer : Tokenizer
{
    public override void Train(string text, int vocabularySize, bool verbose = false) =>
        TrainChunks([TextBytes(text).Select(b => (int)b).ToArray()], vocabularySize, verbose);
    public override int[] Encode(string text) => EncodeChunk(TextBytes(text));
}
