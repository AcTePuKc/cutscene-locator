import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr.model_resolution import ModelResolutionError, validate_model_artifact_layout
from src.asr.readiness import collect_backend_readiness, supported_readiness_backends


class BackendReadinessTests(unittest.TestCase):
    def test_supported_readiness_backends_are_deterministic(self) -> None:
        self.assertEqual(supported_readiness_backends(), ("qwen3-asr", "vibevoice", "whisperx"))

    def test_model_layout_validation_for_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            qwen_dir = Path(temp_dir) / "qwen"
            qwen_dir.mkdir(parents=True)
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "processor_config.json",
                "model.safetensors",
            ):
                (qwen_dir / name).write_text("{}", encoding="utf-8")

            whisperx_dir = Path(temp_dir) / "whisperx"
            whisperx_dir.mkdir(parents=True)
            for name in ("config.json", "model.bin", "tokenizer.json"):
                (whisperx_dir / name).write_text("{}", encoding="utf-8")

            validate_model_artifact_layout(backend_name="qwen3-asr", model_dir=qwen_dir)
            validate_model_artifact_layout(backend_name="whisperx", model_dir=whisperx_dir)

    def test_model_layout_validation_reports_expected_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_qwen = Path(temp_dir) / "missing-qwen"
            missing_qwen.mkdir(parents=True)
            (missing_qwen / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ModelResolutionError, "Resolved qwen3-asr model is missing required artifacts"):
                validate_model_artifact_layout(backend_name="qwen3-asr", model_dir=missing_qwen)

    def test_collect_backend_readiness_reports_registry_and_probe_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            qwen_dir = Path(temp_dir) / "qwen"
            qwen_dir.mkdir(parents=True)
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "processor_config.json",
                "model.safetensors",
            ):
                (qwen_dir / name).write_text("{}", encoding="utf-8")

            with patch("src.asr.readiness.find_spec", return_value=object()):
                with patch("src.asr.registry.find_spec", return_value=object()):
                    row = collect_backend_readiness(backend="qwen3-asr", model_dir=qwen_dir)

        self.assertTrue(row.registry_enabled)
        self.assertEqual(row.missing_dependencies, ())
        self.assertTrue(row.model_layout_valid)
        self.assertEqual(row.cuda_probe_label, "torch")
        self.assertIn("torch CUDA probe", row.cuda_preflight_reason)

    def test_collect_backend_readiness_preserves_missing_dependency_diagnostics(self) -> None:
        def fake_find_spec(name: str):
            if name in {"vibevoice", "torch"}:
                return None
            return object()

        with patch("src.asr.readiness.find_spec", side_effect=fake_find_spec):
            row = collect_backend_readiness(backend="vibevoice", model_dir=None)

        self.assertEqual(row.missing_dependencies, ("vibevoice", "torch"))
        self.assertFalse(row.registry_enabled)
        self.assertFalse(row.model_layout_valid)
        self.assertEqual(row.model_layout_error, "model path not provided")
        self.assertEqual(row.cuda_probe_label, "torch")
        self.assertIn("torch CUDA probe", row.cuda_preflight_reason)


if __name__ == "__main__":
    unittest.main()
