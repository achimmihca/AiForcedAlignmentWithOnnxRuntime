using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;

namespace AiForcedAlignmentWithOnnxRuntime
{
    [TestFixture]
    public class Tests
    {
        private string GetProjectRoot()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            // Try to find the project root by going up until we find TestData or AiForcedAlignmentWithOnnxRuntime.csproj
            DirectoryInfo dir = new DirectoryInfo(baseDir);
            while (dir != null && !File.Exists(Path.Combine(dir.FullName, "AiForcedAlignmentWithOnnxRuntime.csproj")))
            {
                dir = dir.Parent;
            }
            return dir?.FullName ?? baseDir;
        }

        [Test]
        public void TestAlignment()
        {
            string projectRoot = GetProjectRoot();
            string audioPath = Path.Combine(projectRoot, "TestData", "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav");
            string transcriptPath = Path.Combine(projectRoot, "TestData", "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.txt");
            string modelPath = Path.Combine(projectRoot, "AiModels", "distil-wav2vec2_int8.onnx");

            if (!File.Exists(modelPath))
            {
                Assert.Ignore("Model file not found at " + modelPath);
            }

            string transcript = File.ReadAllText(transcriptPath);

            using (Wav2Vec2Aligner aligner = new Wav2Vec2Aligner(modelPath))
            {
                List<WordTimestamp> result = aligner.Align(audioPath, transcript);

                Assert.IsNotEmpty(result);
                foreach (var word in result)
                {
                    Console.WriteLine(word);
                }
            }
        }
    }
}