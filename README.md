# AiForcedAlignmentWithOnnxRuntime
Demo of running a forced aligner model with C# on OnnxRuntime, to find timestamps from given text and audio.

## Prerequisites
- Download the Wav2Vec2 ONNX model and place it `AiModels` folder (see [SOURCE.md](AiModels/SOURCE.md)).

## Implementation Details
- Uses **ONNX Runtime** for model inference.
- Uses **NAudio** for audio loading and resampling (to 16kHz mono).
- Implements **CTC Forced Alignment** (trellis/Viterbi algorithm) in C#.
- Custom `Wav2Vec2Tokenizer` to map characters to model tokens.

## How it works
1. **Audio Preprocessing**: Audio is resampled to 16kHz and converted to mono.
2. **Model Inference**: The audio is passed through the Wav2Vec2 model to get log-probabilities for each character in the vocabulary per 20ms frame.
3. **Forced Alignment**: A trellis is built to find the most likely path that matches the provided transcript.
4. **Timestamp Extraction**: The path is used to find start and end frames for each word.
