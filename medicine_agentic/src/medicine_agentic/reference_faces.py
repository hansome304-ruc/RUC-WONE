from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2


SCHEMA_VERSION = "1.0"
FACE_TYPES = (
    "front_large",
    "back_large",
    "long_side_a",
    "long_side_b",
    "short_side_a",
    "short_side_b",
)
DEFAULT_PICK_FACES = frozenset(("front_large", "back_large"))
MAX_IMAGES_PER_FACE = 3
MIN_IMAGE_SIDE_PX = 32
SUPPORTED_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png"))

_ROOT_KEYS = frozenset(
    (
        "schema_version",
        "bank_id",
        "sku_id",
        "asset_version",
        "status",
        "created_at",
        "units",
        "box_size",
        "object_frame",
        "faces",
        "integrity",
    )
)
_BOX_SIZE_KEYS = frozenset(("length", "width", "height"))
_OBJECT_FRAME_KEYS = frozenset(
    ("origin", "x_axis", "y_axis", "z_axis", "face_normals")
)
_FACE_KEYS = frozenset(("pick_allowed", "images"))
_IMAGE_KEYS = frozenset(("id", "file", "sha256", "width_px", "height_px"))
_INTEGRITY_KEYS = frozenset(("algorithm", "manifest_sha256"))
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EXPECTED_OBJECT_FRAME = {
    "origin": "box_center",
    "x_axis": "length",
    "y_axis": "width",
    "z_axis": "height",
    "face_normals": {
        "front_large": "+z",
        "back_large": "-z",
        "long_side_a": "+y",
        "long_side_b": "-y",
        "short_side_a": "+x",
        "short_side_b": "-x",
    },
}


class ReferenceFaceBankError(RuntimeError):
    """Raised when a reference-face bank cannot be trusted."""


@dataclass(frozen=True)
class ReferenceFace:
    id: str
    face_type: str
    image_path: Path
    pick_allowed: bool
    sha256: str
    width_px: int
    height_px: int


@dataclass(frozen=True)
class ReferenceFaceBank:
    bank_id: str
    content_sha256: str
    faces: tuple[ReferenceFace, ...]
    manifest_path: Path
    sku_id: str
    asset_version: str
    dimensions_mm: tuple[float, float, float]

    def face_by_id(self, face_id: str) -> ReferenceFace:
        matches = [face for face in self.faces if face.id == face_id]
        if len(matches) != 1:
            raise KeyError(face_id)
        return matches[0]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(payload: dict[str, Any]) -> str:
    """Hash canonical JSON, excluding the self-referential manifest hash."""
    candidate = copy.deepcopy(payload)
    integrity = candidate.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("manifest_sha256", None)
    canonical = json.dumps(
        candidate,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _describe_extra_and_missing(
    value: Any,
    *,
    expected: frozenset[str],
    label: str,
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    errors: list[str] = []
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"{label} missing keys: {', '.join(missing)}")
    if extra:
        errors.append(f"{label} has unknown keys: {', '.join(extra)}")
    return errors


def _is_positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) > 0.0
        and float(value) != float("inf")
    )


def _safe_file_path(root: Path, relative_value: Any) -> tuple[Path | None, str | None]:
    if not isinstance(relative_value, str) or not relative_value:
        return None, "file must be a non-empty relative POSIX path"
    if "\\" in relative_value:
        return None, "file must use POSIX '/' separators"
    relative = Path(relative_value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        return None, "file contains an absolute or unsafe path"

    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None, f"symlink is not allowed in reference path: {relative_value}"

    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
    except FileNotFoundError:
        return None, f"reference image does not exist: {relative_value}"
    except (OSError, RuntimeError, ValueError):
        return None, f"reference image escapes bank directory: {relative_value}"
    return resolved_candidate, None


def _decode_image(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return None
    height, width = image.shape[:2]
    if height < MIN_IMAGE_SIDE_PX or width < MIN_IMAGE_SIDE_PX:
        return None
    return int(width), int(height)


def validate_reference_manifest(
    payload: Any,
    manifest_path: str | Path,
    *,
    require_ready: bool = True,
) -> list[str]:
    """Return every validation error without accepting a partially valid bank."""
    path = Path(manifest_path).expanduser()
    root = path.parent
    errors = _describe_extra_and_missing(
        payload,
        expected=_ROOT_KEYS,
        label="manifest",
    )
    if not isinstance(payload, dict):
        return errors

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    for key in ("sku_id", "asset_version"):
        value = payload.get(key)
        if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
            errors.append(f"{key} must match {_SAFE_ID.pattern}")
    expected_bank_id = f"{payload.get('sku_id')}@{payload.get('asset_version')}"
    if payload.get("bank_id") != expected_bank_id:
        errors.append(f"bank_id must be {expected_bank_id!r}")

    status = payload.get("status")
    if status not in ("draft", "ready"):
        errors.append("status must be 'draft' or 'ready'")
    if require_ready and status != "ready":
        errors.append("status must be 'ready'")
    if not isinstance(payload.get("created_at"), str) or not payload.get("created_at"):
        errors.append("created_at must be a non-empty ISO-8601 string")
    if payload.get("units") != "mm":
        errors.append("units must be 'mm'")

    box_size = payload.get("box_size")
    errors.extend(
        _describe_extra_and_missing(
            box_size,
            expected=_BOX_SIZE_KEYS,
            label="box_size",
        )
    )
    if isinstance(box_size, dict):
        for name in _BOX_SIZE_KEYS:
            if not _is_positive_number(box_size.get(name)):
                errors.append(f"box_size.{name} must be a positive finite number")

    object_frame = payload.get("object_frame")
    errors.extend(
        _describe_extra_and_missing(
            object_frame,
            expected=_OBJECT_FRAME_KEYS,
            label="object_frame",
        )
    )
    if isinstance(object_frame, dict) and object_frame != _EXPECTED_OBJECT_FRAME:
        errors.append("object_frame must match the documented carton convention")

    faces = payload.get("faces")
    if not isinstance(faces, dict):
        errors.append("faces must be an object")
        faces = {}
    else:
        actual_faces = set(faces)
        expected_faces = set(FACE_TYPES)
        missing_faces = sorted(expected_faces - actual_faces)
        extra_faces = sorted(actual_faces - expected_faces)
        if missing_faces:
            errors.append(f"faces missing face types: {', '.join(missing_faces)}")
        if extra_faces:
            errors.append(f"faces has unknown face types: {', '.join(extra_faces)}")

    all_ids: set[str] = set()
    all_files: set[str] = set()
    for face_type in FACE_TYPES:
        face = faces.get(face_type)
        errors.extend(
            _describe_extra_and_missing(
                face,
                expected=_FACE_KEYS,
                label=f"faces.{face_type}",
            )
        )
        if not isinstance(face, dict):
            continue
        if not isinstance(face.get("pick_allowed"), bool):
            errors.append(f"faces.{face_type}.pick_allowed must be boolean")
        images = face.get("images")
        if not isinstance(images, list):
            errors.append(f"faces.{face_type}.images must be an array")
            continue
        if len(images) > MAX_IMAGES_PER_FACE:
            errors.append(
                f"faces.{face_type}.images may contain at most "
                f"{MAX_IMAGES_PER_FACE} images"
            )
        if (status == "ready" or require_ready) and face.get("pick_allowed") is True:
            if not 1 <= len(images) <= MAX_IMAGES_PER_FACE:
                errors.append(
                    f"ready bank requires 1-{MAX_IMAGES_PER_FACE} images "
                    f"for pick-allowed face {face_type}"
                )
        for index, record in enumerate(images):
            label = f"faces.{face_type}.images[{index}]"
            errors.extend(
                _describe_extra_and_missing(
                    record,
                    expected=_IMAGE_KEYS,
                    label=label,
                )
            )
            if not isinstance(record, dict):
                continue
            image_id = record.get("id")
            if not isinstance(image_id, str) or not _SAFE_ID.fullmatch(image_id):
                errors.append(f"{label}.id must match {_SAFE_ID.pattern}")
            elif image_id in all_ids:
                errors.append(f"duplicate image id: {image_id}")
            else:
                all_ids.add(image_id)

            relative_file = record.get("file")
            if isinstance(relative_file, str):
                expected_prefix = f"images/{face_type}/"
                if not relative_file.startswith(expected_prefix):
                    errors.append(
                        f"{label}.file must be under {expected_prefix}"
                    )
                if Path(relative_file).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                    errors.append(
                        f"{label}.file must end in .jpg, .jpeg, or .png"
                    )
                if relative_file in all_files:
                    errors.append(f"duplicate reference file: {relative_file}")
                all_files.add(relative_file)
            image_path, path_error = _safe_file_path(root, relative_file)
            if path_error is not None:
                errors.append(f"{label}: {path_error}")
                continue
            assert image_path is not None
            if not image_path.is_file():
                errors.append(f"{label}.file is not a regular file")
                continue

            decoded_size = _decode_image(image_path)
            if decoded_size is None:
                errors.append(
                    f"{label}.file is not a decodable color image of at least "
                    f"{MIN_IMAGE_SIDE_PX}x{MIN_IMAGE_SIDE_PX}px"
                )
            width = record.get("width_px")
            height = record.get("height_px")
            if (
                not isinstance(width, int)
                or isinstance(width, bool)
                or width < MIN_IMAGE_SIDE_PX
            ):
                errors.append(
                    f"{label}.width_px must be an integer >= {MIN_IMAGE_SIDE_PX}"
                )
            if (
                not isinstance(height, int)
                or isinstance(height, bool)
                or height < MIN_IMAGE_SIDE_PX
            ):
                errors.append(
                    f"{label}.height_px must be an integer >= {MIN_IMAGE_SIDE_PX}"
                )
            if decoded_size is not None and decoded_size != (width, height):
                errors.append(
                    f"{label} dimensions do not match decoded image "
                    f"{decoded_size[0]}x{decoded_size[1]}"
                )

            expected_image_hash = record.get("sha256")
            if (
                not isinstance(expected_image_hash, str)
                or not _HEX_SHA256.fullmatch(expected_image_hash)
            ):
                errors.append(f"{label}.sha256 must be a lowercase SHA-256 hex digest")
            else:
                try:
                    actual_image_hash = file_sha256(image_path)
                except OSError as exc:
                    errors.append(f"{label}.file cannot be hashed: {exc}")
                else:
                    if expected_image_hash != actual_image_hash:
                        errors.append(f"{label}.sha256 does not match file content")

    integrity = payload.get("integrity")
    errors.extend(
        _describe_extra_and_missing(
            integrity,
            expected=_INTEGRITY_KEYS,
            label="integrity",
        )
    )
    if isinstance(integrity, dict):
        if integrity.get("algorithm") != "sha256":
            errors.append("integrity.algorithm must be 'sha256'")
        expected_manifest_hash = integrity.get("manifest_sha256")
        if status == "ready" or require_ready:
            if (
                not isinstance(expected_manifest_hash, str)
                or not _HEX_SHA256.fullmatch(expected_manifest_hash)
            ):
                errors.append(
                    "ready bank integrity.manifest_sha256 must be a lowercase "
                    "SHA-256 hex digest"
                )
            else:
                try:
                    actual_manifest_hash = manifest_sha256(payload)
                except (TypeError, ValueError) as exc:
                    errors.append(f"manifest cannot be canonically hashed: {exc}")
                else:
                    if expected_manifest_hash != actual_manifest_hash:
                        errors.append(
                            "integrity.manifest_sha256 does not match the manifest"
                        )
        elif expected_manifest_hash is not None:
            errors.append(
                "draft bank integrity.manifest_sha256 must be null; run finalize "
                "after all images are present"
            )
    return errors


def _read_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser()
    if manifest_path.is_symlink():
        raise ReferenceFaceBankError("manifest.json must not be a symlink")
    if not manifest_path.is_file():
        raise ReferenceFaceBankError(f"manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceFaceBankError(
            f"cannot read manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ReferenceFaceBankError("manifest root must be an object")
    return manifest_path, payload


def load_reference_face_bank(
    path: str | Path,
    *,
    require_ready: bool = True,
) -> ReferenceFaceBank:
    manifest_path, payload = _read_manifest(path)
    errors = validate_reference_manifest(
        payload,
        manifest_path,
        require_ready=require_ready,
    )
    if errors:
        raise ReferenceFaceBankError(
            f"invalid reference face bank {manifest_path}: " + "; ".join(errors)
        )

    root = manifest_path.parent.resolve(strict=True)
    faces: list[ReferenceFace] = []
    for face_type in FACE_TYPES:
        face_record = payload["faces"][face_type]
        for record in face_record["images"]:
            image_path = root.joinpath(*Path(record["file"]).parts).resolve(strict=True)
            faces.append(
                ReferenceFace(
                    id=record["id"],
                    face_type=face_type,
                    image_path=image_path,
                    pick_allowed=face_record["pick_allowed"],
                    sha256=record["sha256"],
                    width_px=record["width_px"],
                    height_px=record["height_px"],
                )
            )
    box_size = payload["box_size"]
    return ReferenceFaceBank(
        bank_id=payload["bank_id"],
        content_sha256=payload["integrity"]["manifest_sha256"] or "",
        faces=tuple(faces),
        manifest_path=manifest_path.resolve(strict=True),
        sku_id=payload["sku_id"],
        asset_version=payload["asset_version"],
        dimensions_mm=(
            float(box_size["length"]),
            float(box_size["width"]),
            float(box_size["height"]),
        ),
    )


def new_reference_manifest(
    *,
    sku_id: str,
    asset_version: str,
    length_mm: float,
    width_mm: float,
    height_mm: float,
    pick_faces: Sequence[str] = tuple(DEFAULT_PICK_FACES),
) -> dict[str, Any]:
    unknown_pick_faces = set(pick_faces) - set(FACE_TYPES)
    if unknown_pick_faces:
        raise ReferenceFaceBankError(
            "unknown pick face types: " + ", ".join(sorted(unknown_pick_faces))
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bank_id": f"{sku_id}@{asset_version}",
        "sku_id": sku_id,
        "asset_version": asset_version,
        "status": "draft",
        "created_at": _utc_now(),
        "units": "mm",
        "box_size": {
            "length": length_mm,
            "width": width_mm,
            "height": height_mm,
        },
        "object_frame": copy.deepcopy(_EXPECTED_OBJECT_FRAME),
        "faces": {
            face_type: {
                "pick_allowed": face_type in pick_faces,
                "images": [],
            }
            for face_type in FACE_TYPES
        },
        "integrity": {
            "algorithm": "sha256",
            "manifest_sha256": None,
        },
    }
    errors = validate_reference_manifest(
        payload,
        Path("manifest.json"),
        require_ready=False,
    )
    missing_file_errors = [
        error for error in errors if "reference image does not exist" in error
    ]
    if errors and len(errors) != len(missing_file_errors):
        raise ReferenceFaceBankError(
            "cannot create invalid draft manifest: " + "; ".join(errors)
        )
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_reference_bank(
    root: str | Path,
    *,
    sku_id: str,
    asset_version: str,
    length_mm: float,
    width_mm: float,
    height_mm: float,
    pick_faces: Sequence[str] = tuple(DEFAULT_PICK_FACES),
) -> Path:
    root_path = Path(root).expanduser()
    version_dir = root_path / sku_id / asset_version
    manifest_path = version_dir / "manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ReferenceFaceBankError(
            f"refusing to overwrite existing manifest: {manifest_path}"
        )
    payload = new_reference_manifest(
        sku_id=sku_id,
        asset_version=asset_version,
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        pick_faces=pick_faces,
    )
    version_dir.mkdir(parents=True, exist_ok=True)
    for face_type in FACE_TYPES:
        (version_dir / "images" / face_type).mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest_path, payload)
    return manifest_path


def add_reference_image(
    manifest: str | Path,
    *,
    face_type: str,
    source_image: str | Path,
) -> dict[str, Any]:
    if face_type not in FACE_TYPES:
        raise ReferenceFaceBankError(f"unknown face type: {face_type}")
    manifest_path, payload = _read_manifest(manifest)
    draft_errors = validate_reference_manifest(
        payload,
        manifest_path,
        require_ready=False,
    )
    if draft_errors:
        raise ReferenceFaceBankError(
            "refusing to edit invalid draft: " + "; ".join(draft_errors)
        )
    if payload["status"] != "draft":
        raise ReferenceFaceBankError(
            "ready bank is immutable; create a new asset_version to change it"
        )
    images = payload["faces"][face_type]["images"]
    if len(images) >= MAX_IMAGES_PER_FACE:
        raise ReferenceFaceBankError(
            f"{face_type} already has {MAX_IMAGES_PER_FACE} reference images"
        )

    source = Path(source_image).expanduser()
    if not source.is_file():
        raise ReferenceFaceBankError(f"source image does not exist: {source}")
    size = _decode_image(source)
    if size is None:
        raise ReferenceFaceBankError(
            f"source must decode as a color image of at least "
            f"{MIN_IMAGE_SIDE_PX}x{MIN_IMAGE_SIDE_PX}px"
        )
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise ReferenceFaceBankError("source image must be .jpg, .jpeg, or .png")

    sequence = len(images) + 1
    image_id = f"{face_type}_{sequence:02d}"
    relative_file = f"images/{face_type}/{image_id}{suffix}"
    destination = manifest_path.parent / relative_file
    if destination.exists() or destination.is_symlink():
        raise ReferenceFaceBankError(f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        image_record = {
            "id": image_id,
            "file": relative_file,
            "sha256": file_sha256(destination),
            "width_px": size[0],
            "height_px": size[1],
        }
        images.append(image_record)
        _atomic_write_json(manifest_path, payload)
        reloaded_path, reloaded = _read_manifest(manifest_path)
        errors = validate_reference_manifest(
            reloaded,
            reloaded_path,
            require_ready=False,
        )
        if errors:
            raise ReferenceFaceBankError(
                "reference bank failed read-back validation: " + "; ".join(errors)
            )
        return image_record
    except Exception:
        if destination.exists():
            destination.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_reference_bank(manifest: str | Path) -> ReferenceFaceBank:
    manifest_path, payload = _read_manifest(manifest)
    if payload.get("status") == "ready":
        return load_reference_face_bank(manifest_path, require_ready=True)
    payload["status"] = "ready"
    integrity = payload.get("integrity")
    if isinstance(integrity, dict):
        integrity["manifest_sha256"] = manifest_sha256(payload)
    errors = validate_reference_manifest(
        payload,
        manifest_path,
        require_ready=True,
    )
    if errors:
        raise ReferenceFaceBankError(
            "cannot finalize reference bank: " + "; ".join(errors)
        )
    _atomic_write_json(manifest_path, payload)
    return load_reference_face_bank(manifest_path, require_ready=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and validate an immutable carton reference-face bank."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init", help="create an empty draft bank")
    initialize.add_argument("--root", default="reference_faces")
    initialize.add_argument("--sku", required=True)
    initialize.add_argument("--version", required=True)
    initialize.add_argument("--length-mm", type=float, required=True)
    initialize.add_argument("--width-mm", type=float, required=True)
    initialize.add_argument("--height-mm", type=float, required=True)
    initialize.add_argument(
        "--pick-face",
        action="append",
        choices=FACE_TYPES,
        help=(
            "face permitted for suction; repeat as needed. "
            "Defaults to front_large and back_large."
        ),
    )

    add = subparsers.add_parser("add", help="copy one image into a draft bank")
    add.add_argument("--manifest", required=True)
    add.add_argument("--face", required=True, choices=FACE_TYPES)
    add.add_argument("--image", required=True)

    finalize = subparsers.add_parser(
        "finalize",
        help=(
            "require every pick-allowed face, hash the manifest, "
            "and make the version immutable"
        ),
    )
    finalize.add_argument("--manifest", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="validate hashes, image decoding, paths, and manifest structure",
    )
    validate.add_argument("--manifest", required=True)
    validate.add_argument(
        "--allow-draft",
        action="store_true",
        help="permit a draft bank with faces not added yet",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            pick_faces = (
                tuple(args.pick_face)
                if args.pick_face is not None
                else tuple(DEFAULT_PICK_FACES)
            )
            manifest_path = initialize_reference_bank(
                args.root,
                sku_id=args.sku,
                asset_version=args.version,
                length_mm=args.length_mm,
                width_mm=args.width_mm,
                height_mm=args.height_mm,
                pick_faces=pick_faces,
            )
            print(f"created draft bank: {manifest_path}")
            return 0
        if args.command == "add":
            record = add_reference_image(
                args.manifest,
                face_type=args.face,
                source_image=args.image,
            )
            print(
                f"added {record['id']}: {record['file']} "
                f"({record['width_px']}x{record['height_px']})"
            )
            return 0
        if args.command == "finalize":
            bank = finalize_reference_bank(args.manifest)
            print(
                f"ready: {bank.bank_id}, {len(bank.faces)} images, "
                f"sha256={bank.content_sha256}"
            )
            return 0
        if args.command == "validate":
            bank = load_reference_face_bank(
                args.manifest,
                require_ready=not args.allow_draft,
            )
            status = "ready" if bank.content_sha256 else "draft"
            print(
                f"valid {status} bank: {bank.bank_id}, "
                f"{len(bank.faces)} images"
            )
            return 0
    except ReferenceFaceBankError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
