import modal


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

