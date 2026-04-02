import json
import tempfile
from pathlib import Path

from audio_utils import extract_chunk_mp3, get_audio_duration_seconds
from modal_app import app, image, model_volume
from segments import dedupe_overlap_segments, normalize_and_offset_segments
from vibevoice_asr import load_vibevoice_asr, transcribe_audio_path


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/model-cache": model_volume},
    timeout=1800,
)
def transcribe_chunked(
    audio_bytes: bytes,
    chunk_seconds: int = 60,
    overlap_seconds: int = 3,
    max_new_tokens: int = 1024,
) -> str:
    if chunk_seconds <= overlap_seconds:
        raise ValueError("chunk_seconds must be greater than overlap_seconds")

    processor, model, device = load_vibevoice_asr()

    with tempfile.TemporaryDirectory() as work_dir:
        audio_path = Path(work_dir) / "input.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        duration = get_audio_duration_seconds(str(audio_path))
        step = chunk_seconds - overlap_seconds
        estimated_chunks = int((duration + step - 1) // step)
        print(
            f"Remote chunking started: duration={duration:.2f}s, "
            f"chunk_seconds={chunk_seconds}, overlap_seconds={overlap_seconds}, "
            f"estimated_chunks={estimated_chunks}"
        )

        all_segments = []
        chunk_index = 0
        start = 0.0

        while start < duration:
            chunk_path = Path(work_dir) / f"chunk_{chunk_index:04d}.mp3"
            extract_chunk_mp3(
                input_path=str(audio_path),
                output_path=str(chunk_path),
                start_seconds=start,
                chunk_seconds=chunk_seconds,
            )

            chunk_segments = transcribe_audio_path(
                processor=processor,
                model=model,
                device=device,
                audio_path=str(chunk_path),
                max_new_tokens=max_new_tokens,
            )
            all_segments.extend(
                normalize_and_offset_segments(chunk_segments, chunk_start_seconds=start)
            )

            print(
                f"Chunk {chunk_index + 1}: start={start:.2f}s, "
                f"segments={len(chunk_segments)}"
            )

            chunk_index += 1
            start += step

    print(
        "Remote chunking complete: "
        f"processed_chunks={chunk_index}, total_segments={len(all_segments)}"
    )
    return json.dumps(dedupe_overlap_segments(all_segments), ensure_ascii=False)

