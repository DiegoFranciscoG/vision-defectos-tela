import io
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image
from pydantic import SecretStr

from fabric_inspection.api.security import ApiKeyVerifier, RateLimiter, key_fingerprint
from fabric_inspection.api.upload import decode_image
from fabric_inspection.data.manifest import (
    ImageRecord,
    assign_splits,
    dhash,
    find_leakage,
    hamming,
    read_manifest,
    scan_dataset,
    split_summary,
    write_manifest,
)
from fabric_inspection.exceptions import ImageTooLargeError, InvalidImageError
from fabric_inspection.inference.visualization import colorize, overlay, to_png_base64
from fabric_inspection.preprocessing import normalize, preprocess, resize_for_model, resize_mask
from fabric_inspection.registry import ModelIntegrityError, resolve_model_file, sha256_file
from tests.conftest import make_registry
from tests.synthetic import make_dataset


def _record(name: str, sha: str, digest: str, split: str) -> ImageRecord:
    return ImageRecord(name, "good", "train", sha, digest, 8, 8, "", split)


class TestManifest:
    def test_scan_split_and_roundtrip(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path, per_defect=5, good_train=16, good_test=4)
        records = assign_splits(scan_dataset(dataset, tmp_path), seed=3)
        summary = split_summary(records)
        assert summary["train"]["good"] == 12
        assert summary["val"]["good"] == 4
        assert summary["test"]["good"] == 4
        assert all(
            summary[split]["hole"] == count
            for split, count in (("train", 2), ("val", 1), ("test", 2))
        )
        assert find_leakage(records) == []
        digest = write_manifest(records, tmp_path / "manifest.csv")
        assert len(digest) == 64
        assert read_manifest(tmp_path / "manifest.csv") == records
        assert assign_splits(records, seed=3) == records  # order-independent and deterministic

    def test_unknown_class_folder_is_rejected(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        (dataset / "train" / "stain").mkdir()
        with pytest.raises(ValueError, match="Unknown class"):
            scan_dataset(dataset, tmp_path)

    def test_leakage_is_reported(self) -> None:
        records = [
            _record("a.png", "1" * 64, "ffff0000ffff0000", "train"),
            _record("b.png", "1" * 64, "0000ffff0000ffff", "test"),
            _record("c.png", "2" * 64, "ffff0000ffff0001", "val"),
        ]
        kinds = {kind for *_, kind in find_leakage(records)}
        assert kinds == {"sha256", "dhash"}

    def test_dhash_and_hamming(self) -> None:
        image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (32, 32), np.uint8))
        assert len(dhash(image)) == 16
        assert hamming(dhash(image), dhash(image)) == 0
        assert hamming("f" * 16, "0" * 16) == 64


class TestPreprocessing:
    def test_shapes_and_normalisation(self) -> None:
        image = Image.new("RGB", (100, 50), (124, 116, 104))
        resized = resize_for_model(image, 64)
        assert resized.size == (64, 64)
        array = normalize(resized)
        assert array.shape == (3, 64, 64)
        assert abs(float(array.mean())) < 0.05  # ImageNet mean colour -> ~0
        assert preprocess(image.convert("L"), 32).shape == (1, 3, 32, 32)
        mask = resize_mask(Image.new("L", (10, 10), 255), 4)
        assert mask.dtype == np.uint8
        assert mask.max() == 1


def _encode(image: Image.Image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class TestUpload:
    def test_valid_png_returns_rgb_and_hash(self) -> None:
        data = _encode(Image.new("L", (40, 40)), "PNG")
        image, digest = decode_image(data, 4096)
        assert image.mode == "RGB"
        assert len(digest) == 64

    @pytest.mark.parametrize(
        ("data", "error"),
        [
            (b"GIF89a....", InvalidImageError),
            (_encode(Image.new("RGB", (40, 40)), "BMP"), InvalidImageError),
            (_encode(Image.new("RGB", (20, 20)), "PNG"), InvalidImageError),
            (_encode(Image.new("RGB", (5000, 40)), "PNG"), ImageTooLargeError),
        ],
    )
    def test_rejected_files(self, data: bytes, error: type[Exception]) -> None:
        with pytest.raises(error):
            decode_image(data, 4096)

    def test_decompression_bomb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
        with pytest.raises(ImageTooLargeError):
            decode_image(_encode(Image.new("RGB", (40, 40)), "PNG"), 4096)

    def test_truncated_png(self) -> None:
        data = _encode(Image.new("RGB", (64, 64), (10, 20, 30)), "PNG")
        with pytest.raises(InvalidImageError):
            decode_image(data[: len(data) // 2], 4096)


class TestSecurity:
    def test_api_key_verifier(self) -> None:
        key = "k" * 40
        verifier = ApiKeyVerifier([SecretStr(key), SecretStr("z" * 40)])
        assert verifier.verify(key) == key_fingerprint(key)
        for bad in (None, "", "k" * 39):
            with pytest.raises(HTTPException) as caught:
                verifier.verify(bad)
            assert caught.value.status_code == 401

    def test_rate_limiter_window(self) -> None:
        limiter = RateLimiter(2, window_s=0.2)
        limiter.check("a")
        limiter.check("a")
        with pytest.raises(HTTPException) as caught:
            limiter.check("a")
        assert caught.value.status_code == 429
        limiter.check("b")  # buckets are independent


class TestVisualizationAndRegistry:
    def test_overlay_png(self) -> None:
        heat = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        colors = colorize(heat)
        assert colors.shape == (64, 64, 3)
        image = overlay(
            Image.new("RGB", (1024, 512)), heat, low=0, high=1, boxes=[(10, 10, 50, 20)]
        )
        assert image.size == (512, 256)
        assert to_png_base64(image).startswith("iVBOR")

    def test_resolve_model_file_prefers_verified_local_copy(self, tmp_path: Path) -> None:
        model = tmp_path / "detector.onnx"
        model.write_bytes(b"onnx-bytes")
        registry = make_registry(detector_sha=sha256_file(model))
        found = resolve_model_file(
            registry.detector, model_dir=tmp_path, cache_dir=tmp_path / "cache", base_url=None
        )
        assert found == model
        model.write_bytes(b"tampered")
        with pytest.raises(ModelIntegrityError, match="not available"):
            resolve_model_file(
                registry.detector, model_dir=tmp_path, cache_dir=tmp_path / "cache", base_url=None
            )
        with pytest.raises(ModelIntegrityError, match="HTTPS"):
            resolve_model_file(
                registry.detector,
                model_dir=tmp_path,
                cache_dir=tmp_path / "cache",
                base_url="http://insecure.example",
            )
