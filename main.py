import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import get_writer
import gc
import os
from dotenv import load_dotenv
import argparse
import sys
import logging
import warnings

# cleaner logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("fsspec").setLevel(logging.ERROR)

# HF_TOKEN
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 用来绕过 torch 2.6 之后的安全警告
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", None)
WAV2VEC_DIR = os.getenv("WAV2VEC2_MODEL_DIR", None)
BATCH_SIZE = 16
COMPUTE_TYPE = "float16"


# 清理显存
def release_memory(model_list):
    for model in model_list:
        if model:
            del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 转写
def transcribe_audio(audio_file, device, lang=None):
    model = whisperx.load_model(
        "large-v3-turbo",
        device,
        compute_type=COMPUTE_TYPE,
        language=lang,
        download_root=MODEL_DIR,
    )
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    print("### LANGUAGE::" + result["language"])
    return model, result, audio


# 对齐
def align_segments(result, audio, device):
    lang = result["language"]
    model_dir = os.path.join(WAV2VEC_DIR, lang)
    model_a, metadata = whisperx.load_align_model(
        language_code=lang,
        device=device,
        model_dir=model_dir,
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    return model_a, result


# 说话人分离
def diarize_speakers(result, audio, device, hf_token):
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
        device=device,
    )
    # min_speakers, max_speakers 可根据需要传入，这里保持 None
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result


# 保存
def save_results(result, audio_file, output_dir):
    output_format = "srt"
    os.makedirs(output_dir, exist_ok=True)
    # 确保 language 字段存在
    if "language" not in result:
        result["language"] = "en"
    writer = get_writer(output_format, output_dir)
    writer(
        result,
        audio_file,
        {
            "highlight_words": False,
            "max_line_count": None,
            "max_line_width": None,
        },
    )
    srt_file_name = (
        os.path.splitext(os.path.basename(audio_file))[0] + "." + output_format
    )
    srt_file_path = os.path.join(output_dir, srt_file_name)
    # 重要, 用于最终给 n8n 工作流里传参. 需要把 \ 转换成 /
    print(f"### OUTPUT_SRT_PATH::{os.path.abspath(srt_file_path).replace('\\', '/')}")


def main():
    # 设置 args
    parser = argparse.ArgumentParser(description="WhisperX Processing Script")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument(
        "--output_dir", help="Directory to save the SRT file", default=None
    )
    parser.add_argument("--lang", help="Language code (e.g. en, zh)", default=None)
    parser.add_argument(
        "-D",
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (Requires HF_TOKEN)",
    )
    parser.add_argument(
        "-N",
        "--no_release_memory",
        action="store_true",
        help="Disable VRAM cleanup to speed up consecutive runs (Risk of OOM)",
    )
    args = parser.parse_args()

    audio_file = args.input_file
    if not os.path.exists(audio_file):
        print(f"Error: File not found {audio_file}")
        sys.exit(1)
    output_dir = args.output_dir if args.output_dir else os.path.dirname(audio_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"### Processing: {audio_file} on {device}")

    model_whisper, result, audio = transcribe_audio(audio_file, device, args.lang)
    model_align, result = align_segments(result, audio, device)
    if not args.no_release_memory:
        release_memory([model_whisper, model_align])
        model_whisper = None
        model_align = None
    if args.diarize:
        result = diarize_speakers(result, audio, device, HF_TOKEN)
    save_results(result, audio_file, output_dir)


if __name__ == "__main__":
    main()
