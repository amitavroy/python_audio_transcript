from __future__ import annotations

from typing import Any, Dict, List


def load_vibevoice_asr(*, hf_home: str = "/model-cache", model_path: str = "microsoft/VibeVoice-ASR"):
    import os
    import torch
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    os.environ["HF_HOME"] = hf_home

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
    return processor, model, device


def transcribe_audio_path(
    *,
    processor: Any,
    model: Any,
    device: Any,
    audio_path: str,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    import torch

    inputs = processor(
        audio=audio_path,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

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
    segments = processor.post_process_transcription(generated_text)
    if segments:
        return segments
    return [{"Start": 0.0, "End": 0.0, "Speaker": -1, "Content": generated_text}]

