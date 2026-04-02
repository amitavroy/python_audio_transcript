import subprocess
from pathlib import Path


def get_audio_duration_seconds(audio_path: str) -> float:
    duration = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        text=True,
    ).strip()
    return float(duration)


def extract_chunk_mp3(
    *,
    input_path: str,
    output_path: str,
    start_seconds: float,
    chunk_seconds: int,
    sample_rate_hz: int = 16000,
    channels: int = 1,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_seconds}",
            "-t",
            f"{chunk_seconds}",
            "-i",
            input_path,
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate_hz),
            output_path,
        ],
        check=True,
    )

