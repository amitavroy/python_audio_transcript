import modal
import json
import ast
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

app = modal.App()

model_volume = modal.Volume.from_name("vibevoice-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .run_commands(
        "git clone https://github.com/microsoft/VibeVoice.git /opt/vibevoice",
        "pip install -e /opt/vibevoice",
    )
    .pip_install("torch", "transformers", "accelerate")
)


def _get_audio_duration_seconds(audio_path: str) -> float:
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


def _split_audio_chunks(audio_path: str, chunk_seconds: int, overlap_seconds: int):
    if chunk_seconds <= overlap_seconds:
        raise ValueError("chunk_seconds must be greater than overlap_seconds")

    temp_dir = tempfile.TemporaryDirectory()
    output_dir = Path(temp_dir.name)
    duration = _get_audio_duration_seconds(audio_path)
    step = chunk_seconds - overlap_seconds

    chunks = []
    chunk_index = 0
    start = 0.0
    while start < duration:
        chunk_path = output_dir / f"chunk_{chunk_index:04d}.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start}",
                "-t",
                f"{chunk_seconds}",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                str(chunk_path),
            ],
            check=True,
        )
        chunks.append((start, chunk_path))
        chunk_index += 1
        start += step

    return temp_dir, chunks


def _parse_segments(raw: str):
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Backward compatible parse for stringified Python lists.
        parsed = ast.literal_eval(raw)

    if isinstance(parsed, list):
        return parsed
    return [{"Start": 0.0, "End": 0.0, "Speaker": -1, "Content": str(raw)}]


def _dedupe_overlap_segments(segments):
    deduped = []
    for current in segments:
        if deduped:
            previous = deduped[-1]
            prev_text = str(previous.get("Content") or previous.get("text") or "").strip()
            curr_text = str(current.get("Content") or current.get("text") or "").strip()
            same_text = prev_text == curr_text and bool(curr_text)
            prev_start = float(previous.get("Start", previous.get("start_time", 0.0)))
            curr_start = float(current.get("Start", current.get("start_time", 0.0)))
            near_boundary = abs(curr_start - prev_start) < 4.0
            if same_text and near_boundary:
                continue
        deduped.append(current)
    return deduped


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
    import os
    import json
    import torch
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    if chunk_seconds <= overlap_seconds:
        raise ValueError("chunk_seconds must be greater than overlap_seconds")

    os.environ["HF_HOME"] = "/model-cache"
    model_path = "microsoft/VibeVoice-ASR"

    processor = VibeVoiceASRProcessor.from_pretrained(
        model_path,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    device = next(model.parameters()).device

    with tempfile.TemporaryDirectory() as work_dir:
        audio_path = Path(work_dir) / "input.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        duration = _get_audio_duration_seconds(str(audio_path))
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
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start}",
                    "-t",
                    f"{chunk_seconds}",
                    "-i",
                    str(audio_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(chunk_path),
                ],
                check=True,
            )

            inputs = processor(
                audio=str(chunk_path),
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=processor.pad_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            generated_text = processor.decode(generated_ids, skip_special_tokens=True)
            chunk_segments = processor.post_process_transcription(generated_text) or [
                {"Start": 0.0, "End": 0.0, "Speaker": -1, "Content": generated_text}
            ]

            for seg in chunk_segments:
                raw_start = seg.get("Start", seg.get("start_time", 0.0))
                raw_end = seg.get("End", seg.get("end_time", raw_start))
                seg_start = float(raw_start)
                seg_end = float(raw_end)
                absolute_start = round(start + seg_start, 2)
                absolute_end = round(start + seg_end, 2)

                # Normalize fields so downstream logic has consistent keys.
                seg["Start"] = absolute_start
                seg["End"] = absolute_end
                seg["start_time"] = absolute_start
                seg["end_time"] = absolute_end
                if "Content" not in seg and "text" in seg:
                    seg["Content"] = seg["text"]
                if "text" not in seg and "Content" in seg:
                    seg["text"] = seg["Content"]
            all_segments.extend(chunk_segments)
            print(
                f"Chunk {chunk_index + 1}: start={start:.2f}s, "
                f"segments={len(chunk_segments)}"
            )

            chunk_index += 1
            start += step

    print(f"Remote chunking complete: processed_chunks={chunk_index}, total_segments={len(all_segments)}")
    return json.dumps(_dedupe_overlap_segments(all_segments), ensure_ascii=False)


@app.local_entrypoint()
def main(
    audio_path: str = "audio.mp3",
    chunk_seconds: int = 60,
    overlap_seconds: int = 1,
    max_new_tokens: int = 1024,
    yes: bool = False,
    confirm: str = "",
):
    import sys

    resolved_audio_path = str(Path(audio_path).expanduser().resolve())
    if not Path(resolved_audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_audio_path}")

    start = datetime.now()
    print(f"Start time: {start.strftime('%H:%M:%S')}")
    print(f"Using audio file: {resolved_audio_path}")
    if not yes:
        confirmation = confirm.strip().lower()
        if not confirmation:
            if sys.stdin.isatty():
                try:
                    confirmation = input("Proceed with this audio file? [y/N]: ").strip().lower()
                except EOFError:
                    confirmation = ""
            else:
                raise RuntimeError(
                    "Interactive confirmation is unavailable in this terminal. "
                    "Re-run with --confirm y (or --yes) to proceed."
                )
        if confirmation not in {"y", "yes"}:
            print("Cancelled by user.")
            return

    with open(resolved_audio_path, "rb") as f:
        result = transcribe_chunked.remote(
            f.read(),
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            max_new_tokens=max_new_tokens,
        )

    end = datetime.now()
    elapsed = end - start
    print(f"End time:   {end.strftime('%H:%M:%S')}")
    print(f"Elapsed:    {str(elapsed).split('.')[0]}")
    print(result)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"output_{timestamp}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Wrote transcript to: {output_path}")
