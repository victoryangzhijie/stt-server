from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {
        "env_prefix": "STT_",
        "env_file": Path(__file__).parent / ".env",
        "env_file_encoding": "utf-8",
    }

    backend: str = "qwen3"
    host: str = "0.0.0.0"
    port: int = 8000

    # Queue sizes
    in_queue_size: int = 200
    out_queue_size: int = 50
    in_queue_max_drops: int = 500  # cumulative dropped frames before disconnect (500×20ms=10s)

    # File upload limits
    max_upload_bytes: int = 100 * 1024 * 1024  # 100MB

    # Supported audio params (for validation)
    supported_sample_rates: list[int] = [16000]
    supported_codecs: list[str] = ["pcm_s16le"]
    supported_channels: list[int] = [1]

    # Logging config
    log_dir: Path = Path("logs")

    # vLLM backend config
    vllm_model: str = "Qwen/Qwen3-ASR-1.7B"
    vllm_gpu_memory: float = 0.7  # Reduced from 0.8 for more KV cache headroom
    vllm_max_tokens: int = 512
    vllm_max_inference_batch_size: int = 32  # Prevent OOM on batch requests

    # VAD config
    vad_threshold: int = 300
    vad_silence_ms: int = 500  # 降低延迟: 700ms -> 500ms

    # qwen-asr streaming config
    streaming_chunk_size_sec: float = 0.5  # 降低延迟: 1.0s -> 0.5s
    streaming_unfixed_chunk_num: int = 2
    streaming_unfixed_token_num: int = 5

    # Whisper backend config
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "float16"
    whisper_beam_size: int = 5
    whisper_vad_threshold: float = 0.5


settings = Settings()
