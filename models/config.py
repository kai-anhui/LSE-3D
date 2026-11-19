from dataclasses import dataclass


@dataclass
class ModelConfig:
    dim: int = 256
    hidden: int = 128
    state: int = 16
    gate_hidden: int = 128
    gate_mode: str = "recurrent"
    gate_use_language: bool = True
    states: str = "dual"
    directions: int = 4
    scan_backend: str = "chunked"
    text_model: str = ""
    freeze_text: bool = False
    max_tokens: int = 256
