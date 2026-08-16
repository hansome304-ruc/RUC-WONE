from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from medicine_agentic.reference_faces import (
    FACE_TYPES,
    ReferenceFaceBankError,
    add_reference_image,
    finalize_reference_bank,
    initialize_reference_bank,
    load_reference_face_bank,
    manifest_sha256,
    validate_reference_manifest,
)


def write_image(path: Path, value: int = 127) -> None:
    image = np.full((64, 96, 3), value, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to create test image: {path}")


class ReferenceFaceBankTests(unittest.TestCase):
    def create_draft(self, directory: Path) -> Path:
        return initialize_reference_bank(
            directory / "reference_faces",
            sku_id="medicine_carton_001",
            asset_version="1.0.0",
            length_mm=120.0,
            width_mm=70.0,
            height_mm=20.0,
        )

    def populate_all_faces(self, manifest: Path, directory: Path) -> None:
        for index, face_type in enumerate(FACE_TYPES):
            source = directory / f"{face_type}.png"
            write_image(source, 40 + index * 20)
            add_reference_image(
                manifest,
                face_type=face_type,
                source_image=source,
            )

    def test_draft_requires_every_face_before_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.create_draft(directory)
            source = directory / "front.png"
            write_image(source)
            add_reference_image(
                manifest,
                face_type="front_large",
                source_image=source,
            )
            with self.assertRaisesRegex(
                ReferenceFaceBankError,
                "requires 1-3 images for pick-allowed face back_large",
            ):
                finalize_reference_bank(manifest)
            with self.assertRaisesRegex(
                ReferenceFaceBankError,
                "status must be 'ready'",
            ):
                load_reference_face_bank(manifest)

    def test_finalize_and_load_builds_typed_bank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.create_draft(directory)
            self.populate_all_faces(manifest, directory)
            bank = finalize_reference_bank(manifest)

            self.assertEqual(bank.bank_id, "medicine_carton_001@1.0.0")
            self.assertEqual(bank.dimensions_mm, (120.0, 70.0, 20.0))
            self.assertEqual(len(bank.faces), 6)
            self.assertEqual(len(bank.content_sha256), 64)
            self.assertTrue(bank.face_by_id("front_large_01").pick_allowed)
            self.assertFalse(bank.face_by_id("long_side_a_01").pick_allowed)

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["integrity"]["manifest_sha256"],
                manifest_sha256(payload),
            )
            with self.assertRaisesRegex(ReferenceFaceBankError, "immutable"):
                add_reference_image(
                    manifest,
                    face_type="front_large",
                    source_image=directory / "front_large.png",
                )

    def test_task1_bank_can_finalize_with_only_two_pick_faces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.create_draft(directory)
            for index, face_type in enumerate(("front_large", "back_large")):
                source = directory / f"{face_type}.png"
                write_image(source, 80 + index * 40)
                add_reference_image(
                    manifest,
                    face_type=face_type,
                    source_image=source,
                )

            bank = finalize_reference_bank(manifest)

            self.assertEqual(len(bank.faces), 2)
            self.assertEqual(
                {face.face_type for face in bank.faces},
                {"front_large", "back_large"},
            )

    def test_modified_image_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.create_draft(directory)
            self.populate_all_faces(manifest, directory)
            bank = finalize_reference_bank(manifest)
            write_image(bank.face_by_id("front_large_01").image_path, value=255)

            with self.assertRaisesRegex(
                ReferenceFaceBankError,
                "sha256 does not match file content",
            ):
                load_reference_face_bank(manifest)

    def test_unknown_key_and_manifest_hash_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.create_draft(directory)
            self.populate_all_faces(manifest, directory)
            finalize_reference_bank(manifest)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["operator_note"] = "not permitted"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ReferenceFaceBankError,
                "unknown keys: operator_note",
            ):
                load_reference_face_bank(manifest)

            payload.pop("operator_note")
            payload["box_size"]["length"] = 121.0
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ReferenceFaceBankError,
                "manifest_sha256 does not match",
            ):
                load_reference_face_bank(manifest)

    def test_traversal_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.create_draft(directory)
            source = directory / "source.png"
            write_image(source)
            record = add_reference_image(
                manifest,
                face_type="front_large",
                source_image=source,
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["faces"]["front_large"]["images"][0]["file"] = "../../source.png"
            errors = validate_reference_manifest(
                payload,
                manifest,
                require_ready=False,
            )
            self.assertTrue(any("unsafe path" in error for error in errors), errors)

            payload["faces"]["front_large"]["images"][0]["file"] = record["file"]
            destination = manifest.parent / record["file"]
            destination.unlink()
            try:
                os.symlink(source, destination)
            except (OSError, NotImplementedError):
                self.skipTest("filesystem does not support symlinks")
            errors = validate_reference_manifest(
                payload,
                manifest,
                require_ready=False,
            )
            self.assertTrue(any("symlink is not allowed" in error for error in errors), errors)

    def test_bad_dimensions_and_undecodable_image_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(
                ReferenceFaceBankError,
                "box_size.length must be a positive",
            ):
                initialize_reference_bank(
                    directory,
                    sku_id="carton",
                    asset_version="1",
                    length_mm=0,
                    width_mm=70,
                    height_mm=20,
                )

            manifest = self.create_draft(directory)
            bad_image = directory / "bad.jpg"
            bad_image.write_bytes(b"not an image")
            with self.assertRaisesRegex(ReferenceFaceBankError, "must decode"):
                add_reference_image(
                    manifest,
                    face_type="front_large",
                    source_image=bad_image,
                )


if __name__ == "__main__":
    unittest.main()
