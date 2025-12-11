# WhisperX-easy-subtitle

A streamlined, n8n-friendly CLI wrapper for [WhisperX](https://github.com/m-bain/whisperX). It provides fast speech-to-text transcription with accurate timestamps and speaker diarization, optimized for automation workflows.

## Features

*   🚀 **Fast Transcription:** Uses Faster-Whisper (large-v3-turbo).
*   👥 **Speaker Diarization:** Identifies different speakers via Pyannote 3.1.
*   🧠 **Smart Memory Management:** Automatic VRAM cleanup (configurable).
*   🔌 **Automation Ready:** Outputs standardized paths for n8n/workflow integration.
*   💾 **Customizable Storage:** Supports custom model paths via environment variables.

## Prerequisites

*   `uv`
*   `FFmpeg` (Required for audio processing)
*   NVIDIA GPU + CUDA (Highly Recommended)

## Installation

1.  **Clone the repository**
    ```sh
    git clone https://github.com/ryong5688/WhisperX-easy-subtitle.git
    cd WhisperX-easy-subtitle
    ```

2.  **Install dependencies**
    ```sh
    uv sync
    ```

## Configuration

1.  Create a `.env` file in the root directory:

2.  Add your Hugging Face Token (Required for Diarization):
    ```env
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    # Optional: Custom model cache paths
    # WHISPER_MODEL_DIR=./models/whisper
    # WAV2VEC2_MODEL_DIR=./models/wav2vec
    ```

    > **Note:** You must accept the user agreements for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on HuggingFace.

## Usage

Run the script using `uv run`.

### Basic Transcription
```sh
uv run main.py input.wav
```

### With Speaker Diarization
Add the `-D` flag. Requires `HF_TOKEN`.
```sh
uv run main.py interview.mp3 -D
```

### Full Argument List

| Argument        | Flag                        | Description                                                    |
| :-------------- | :-------------------------- | :------------------------------------------------------------- |
| **Input**       | `input_file`                | Path to the audio file (wav, mp3, etc.)                        |
| **Output**      | `--output_dir`              | Directory to save the `.srt` file (Default: same as input)     |
| **Language**    | `--lang`                    | Language code (e.g., `en`, `zh`). Auto-detects if omitted.     |
| **Diarization** | `-D`, `--diarize`           | Enable speaker identification.                                 |
| **Memory**      | `-N`, `--no_release_memory` | Keep models in VRAM. Speeds up consecutive runs but risks OOM. |

## Automation (n8n)

The script prints the absolute path of the generated SRT file to `stdout` for easy parsing:

```text
### OUTPUT_SRT_PATH::/abs/path/to/your/file.srt
```

## License

MIT License. Based on [WhisperX](https://github.com/m-bain/whisperX).
