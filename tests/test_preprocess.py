import subprocess
import tempfile
import unittest
from pathlib import Path

from src.ingest.preprocess import preprocess_media


class PreprocessTests(unittest.TestCase):
    def test_preprocess_chunk_disabled_uses_canonical_only(self) -> None:
        commands: list[list[str]] = []

        def fake_runner(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            result = preprocess_media(
                ffmpeg_binary="ffmpeg",
                input_path=Path("Mission 01.mp4"),
                out_dir=out_dir,
                chunk_seconds=0,
                runner=fake_runner,
            )

            self.assertEqual(len(commands), 1)
            self.assertTrue(str(result.canonical_wav_path).endswith("mission_01_canonical.wav"))
            self.assertEqual(len(result.chunk_metadata), 1)
            self.assertEqual(result.chunk_metadata[0].chunk_index, 0)
            self.assertEqual(result.chunk_metadata[0].absolute_offset_seconds, 0.0)
            self.assertEqual(result.chunk_metadata[0].chunk_wav_path, result.canonical_wav_path)
            self.assertTrue((out_dir / "_tmp" / "audio").exists())
            self.assertTrue((out_dir / "_tmp" / "chunks").exists())

    def test_preprocess_chunking_emits_offsets(self) -> None:
        def fake_runner(command, **kwargs):
            if "-f" in command and "segment" in command:
                chunk_pattern = Path(command[-1])
                chunk_pattern.parent.mkdir(parents=True, exist_ok=True)
                (chunk_pattern.parent / f"{chunk_pattern.stem.replace('%06d', '000000')}.wav").touch()
                (chunk_pattern.parent / f"{chunk_pattern.stem.replace('%06d', '000001')}.wav").touch()
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            result = preprocess_media(
                ffmpeg_binary="ffmpeg",
                input_path=Path("in.wav"),
                out_dir=out_dir,
                chunk_seconds=300,
                runner=fake_runner,
            )

            self.assertEqual(len(result.chunk_metadata), 2)
            self.assertEqual(result.chunk_metadata[0].absolute_offset_seconds, 0.0)
            self.assertEqual(result.chunk_metadata[1].absolute_offset_seconds, 300.0)
            self.assertEqual(result.chunk_metadata[0].source_filename, "in.wav")

    def test_negative_chunk_seconds_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                preprocess_media(
                    ffmpeg_binary="ffmpeg",
                    input_path=Path("in.wav"),
                    out_dir=Path(temp_dir),
                    chunk_seconds=-1,
                )


if __name__ == "__main__":
    unittest.main()
