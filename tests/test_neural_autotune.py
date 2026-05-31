from persona_chess.neural import HardwareProfile, recommend_neural_config


def test_auto_config_prefers_cpu_safe_defaults_without_cuda() -> None:
    hardware = HardwareProfile(
        cpu_count=8,
        memory_gb=16.0,
        torch_available=False,
        cuda_available=False,
        cuda_device_name=None,
        cuda_memory_gb=None,
        device_type="cpu",
    )

    config = recommend_neural_config(10_000, hardware=hardware)

    assert config.profile == "small"
    assert config.transformer.d_model == 128
    assert config.training.batch_size == 16
    assert config.training.gradient_accumulation_steps == 4
    assert config.effective_batch_size == 64
    assert config.training.epochs == 5
    assert config.lora.rank == 4


def test_auto_config_uses_larger_lora_profile_on_high_memory_cuda() -> None:
    hardware = HardwareProfile(
        cpu_count=16,
        memory_gb=64.0,
        torch_available=True,
        cuda_available=True,
        cuda_device_name="Test GPU",
        cuda_memory_gb=24.0,
        device_type="cuda",
    )

    config = recommend_neural_config(2_000_000, hardware=hardware)

    assert config.profile == "large"
    assert config.transformer.d_model == 384
    assert config.training.batch_size == 96
    assert config.effective_batch_size >= 256
    assert config.training.epochs == 2
    assert config.lora.rank == 16


def test_auto_config_respects_explicit_profile() -> None:
    hardware = HardwareProfile(
        cpu_count=4,
        memory_gb=8.0,
        torch_available=False,
        cuda_available=False,
        cuda_device_name=None,
        cuda_memory_gb=None,
        device_type="cpu",
    )

    config = recommend_neural_config(100_000, profile="balanced", hardware=hardware)

    assert config.profile == "balanced"
    assert config.transformer.n_layers == 4
    assert "explicitly requested" in " ".join(config.notes)
