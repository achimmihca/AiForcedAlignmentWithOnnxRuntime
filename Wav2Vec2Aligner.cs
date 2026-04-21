using System;
using System.Collections.Generic;
using System.Linq;
using NAudio.Wave;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AiForcedAlignmentWithOnnxRuntime
{
    public interface IWav2Vec2Model : IDisposable
    {
        float[,,] RunInference(float[] audioData);
    }

    public class OnnxWav2Vec2Model : IWav2Vec2Model
    {
        private readonly InferenceSession _session;

        public OnnxWav2Vec2Model(string modelPath)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };
            
            // For Desktop CPU, optimizing thread count can help. 
            // By default ORT uses all cores, but sometimes limiting to physical cores is better.
            // options.IntraOpNumThreads = Environment.ProcessorCount; 

            _session = new InferenceSession(modelPath, options);
        }

        public float[,,] RunInference(float[] audioData)
        {
            var tensor = new DenseTensor<float>(audioData, new[] { 1, audioData.Length });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_values", tensor)
            };

            using (var results = _session.Run(inputs))
            {
                var outputTensor = results.First().AsTensor<float>();
                int numFrames = outputTensor.Dimensions[1];
                int numTokens = outputTensor.Dimensions[2];
                
                float[,,] logProbs = new float[1, numFrames, numTokens];
                for (int i = 0; i < numFrames; i++)
                {
                    for (int j = 0; j < numTokens; j++)
                    {
                        logProbs[0, i, j] = outputTensor[0, i, j];
                    }
                }
                return logProbs;
            }
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    public class Wav2Vec2Aligner : IDisposable
    {
        private readonly IWav2Vec2Model _model;
        private readonly Wav2Vec2Tokenizer _tokenizer;

        public Wav2Vec2Aligner(string modelPath) : this(new OnnxWav2Vec2Model(modelPath))
        {
        }

        public Wav2Vec2Aligner(IWav2Vec2Model model)
        {
            _model = model;
            _tokenizer = new Wav2Vec2Tokenizer();
        }

        public List<WordTimestamp> Align(string audioPath, string transcript)
        {
            float[] audioData = LoadAudio(audioPath);
            var logProbs = _model.RunInference(audioData);
            
            string normalizedTranscript = _tokenizer.Normalize(transcript);
            int[] tokenIds = _tokenizer.Encode(normalizedTranscript);

            return FindAlignment(logProbs, tokenIds, normalizedTranscript);
        }

        private float[] LoadAudio(string path)
        {
            using (var reader = new AudioFileReader(path))
            {
                var sampleProvider = reader.ToSampleProvider();
                
                // Resample to 16kHz
                if (reader.WaveFormat.SampleRate != 16000)
                {
                    // WdlResamplingSampleProvider is a good built-in resampler in NAudio
                    sampleProvider = new NAudio.Wave.SampleProviders.WdlResamplingSampleProvider(sampleProvider, 16000);
                }
                
                // Convert to Mono
                if (sampleProvider.WaveFormat.Channels != 1)
                {
                    sampleProvider = sampleProvider.ToMono();
                }

                List<float> samples = new List<float>();
                float[] readBuffer = new float[16000];
                int samplesRead;
                
                while ((samplesRead = sampleProvider.Read(readBuffer, 0, readBuffer.Length)) > 0)
                {
                    samples.AddRange(readBuffer.Take(samplesRead));
                }
                return samples.ToArray();
            }
        }


        private List<WordTimestamp> FindAlignment(float[,,] logProbs, int[] tokenIds, string normalizedTranscript)
        {
            int numFrames = logProbs.GetLength(1);
            int numTokens = logProbs.GetLength(2);
            int targetLen = tokenIds.Length;

            // Simple CTC alignment (greedy for now just to see if it works, 
            // but we need Forced Alignment trellis)
            
            // Trellis: [num_frames + 1, target_len + 1]
            float[,] trellis = new float[numFrames + 1, targetLen + 1];
            for (int i = 0; i <= numFrames; i++)
                for (int j = 0; j <= targetLen; j++)
                    trellis[i, j] = float.NegativeInfinity;

            trellis[0, 0] = 0;
            for (int t = 1; t <= numFrames; t++)
            {
                for (int s = 0; s <= targetLen; s++)
                {
                    // Stay in same state
                    float stay = trellis[t - 1, s];
                    // Move from previous state
                    float move = s > 0 ? trellis[t - 1, s - 1] : float.NegativeInfinity;
                    
                    float prob = s > 0 ? logProbs[0, t - 1, tokenIds[s - 1]] : logProbs[0, t - 1, 0]; // 0 is usually blank
                    
                    trellis[t, s] = Math.Max(stay, move) + prob;
                }
            }

            // Backtracking
            List<int> path = new List<int>();
            int currS = targetLen;
            for (int t = numFrames; t > 0; t--)
            {
                path.Add(currS);
                if (currS > 0)
                {
                    float stay = trellis[t - 1, currS];
                    float move = trellis[t - 1, currS - 1];
                    if (move > stay)
                    {
                        currS--;
                    }
                }
            }
            path.Reverse();

            // Extract timestamps
            // Wav2Vec2 frames are 20ms apart (50Hz) or similar depending on the model.
            // For wav2vec2-base-960h, it's 16kHz input, and output is downsampled by 320x.
            // So each frame is 320 / 16000 = 0.02 seconds = 20ms.
            double frameDuration = 0.02; 

            List<WordTimestamp> words = new List<WordTimestamp>();
            string[] rawWords = normalizedTranscript.Split('|');
            int tokenIdx = 0;
            
            for (int i = 0; i < rawWords.Length; i++)
            {
                string word = rawWords[i];
                int startFrame = -1;
                int endFrame = -1;

                // Find tokens for this word
                for (int j = 0; j < word.Length; j++)
                {
                    // Find first frame where path[t] == tokenIdx + 1
                    for (int t = 0; t < path.Count; t++)
                    {
                        if (path[t] == tokenIdx + 1)
                        {
                            if (startFrame == -1) startFrame = t;
                            endFrame = t;
                        }
                    }
                    tokenIdx++;
                }
                tokenIdx++; // Skip space token

                if (startFrame != -1)
                {
                    words.Add(new WordTimestamp
                    {
                        Word = word,
                        Start = startFrame * frameDuration,
                        End = (endFrame + 1) * frameDuration
                    });
                }
            }

            return words;
        }

        public void Dispose()
        {
            _model?.Dispose();
        }
    }

    public class WordTimestamp
    {
        public string Word { get; set; }
        public double Start { get; set; }
        public double End { get; set; }
        public override string ToString() => $"{Word}: {Start:F2} - {End:F2}";
    }

    public class Wav2Vec2Tokenizer
    {
        // Simple tokenizer for Wav2Vec2-base-960h
        // Alphabet: [blank], <pad>, <s>, </s>, <unk>, |, E, T, A, O, N, I, H, S, R, D, L, U, M, W, C, F, G, Y, P, B, V, K, J, X, Q, Z
        // Usually | is space (index 4)
        // Wait, standard wav2vec2-960h vocab:
        // 0: <pad>, 1: <s>, 2: </s>, 3: <unk>, 4: |, 5: E, 6: T, ...
        private static readonly char[] Vocab = "    |ETAONIHSRDLUMWCFGYPBVKJXQZ".ToCharArray();

        public string Normalize(string text)
        {
            return text.ToUpper().Replace(" ", "|");
        }

        public int[] Encode(string normalizedText)
        {
            List<int> ids = new List<int>();
            foreach (char c in normalizedText)
            {
                int id = Array.IndexOf(Vocab, c);
                if (id == -1) id = 3; // <unk>
                ids.Add(id);
            }
            return ids.ToArray();
        }
    }
}
