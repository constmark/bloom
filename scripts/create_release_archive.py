#!/usr/bin/env python3
"""Create a deterministic Bloom release tar.gz or zip archive."""

from __future__ import annotations

import argparse
import gzip
import os
import pathlib
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import zipfile


MAX_SOURCE_DATE_EPOCH = (1 << 32) - 1
ZIP_EPOCH = 315_532_800
EPOCH_RE = re.compile(r"^(0|[1-9][0-9]*)$")


class ArchiveError(Exception):
    """The source tree or requested archive metadata is invalid."""


def parse_source_date_epoch(raw: str) -> int:
    if EPOCH_RE.fullmatch(raw) is None:
        raise ArchiveError("SOURCE_DATE_EPOCH must be an unsigned decimal integer")
    epoch = int(raw)
    if epoch > MAX_SOURCE_DATE_EPOCH:
        raise ArchiveError(
            f"SOURCE_DATE_EPOCH must not exceed {MAX_SOURCE_DATE_EPOCH}"
        )
    return epoch


def archive_mode(path: pathlib.Path) -> int:
    if path.is_dir():
        return 0o755
    return 0o755 if path.stat().st_mode & 0o111 else 0o644


def source_entries(package_root: pathlib.Path) -> list[pathlib.Path]:
    if package_root.is_symlink() or not package_root.is_dir():
        raise ArchiveError("package root must be a real directory")
    entries = [package_root, *package_root.rglob("*")]
    entries.sort(key=lambda path: path.relative_to(package_root.parent).as_posix())
    for path in entries:
        if path.is_symlink():
            raise ArchiveError(f"symbolic links are not allowed: {path}")
        if not (path.is_dir() or path.is_file()):
            raise ArchiveError(f"special filesystem entries are not allowed: {path}")
    return entries


def create_tar_gz(
    package_root: pathlib.Path,
    destination: pathlib.Path,
    entries: list[pathlib.Path],
    epoch: int,
) -> None:
    with destination.open("wb") as raw_archive:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw_archive,
            mtime=epoch,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
            ) as archive:
                for path in entries:
                    member_name = path.relative_to(package_root.parent).as_posix()
                    info = archive.gettarinfo(str(path), arcname=member_name)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = epoch
                    info.mode = archive_mode(path)
                    info.pax_headers = {}
                    if path.is_file():
                        with path.open("rb") as source:
                            archive.addfile(info, source)
                    else:
                        archive.addfile(info)


def create_zip(
    package_root: pathlib.Path,
    destination: pathlib.Path,
    entries: list[pathlib.Path],
    epoch: int,
) -> None:
    zip_timestamp = time.gmtime(max(epoch, ZIP_EPOCH))[:6]
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for path in entries:
            if not path.is_file():
                continue
            member_name = path.relative_to(package_root.parent).as_posix()
            info = zipfile.ZipInfo(member_name, date_time=zip_timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | archive_mode(path)) << 16
            with path.open("rb") as source, archive.open(info, mode="w") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)


def create_archive(
    package_root: pathlib.Path, destination: pathlib.Path, epoch: int
) -> None:
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not (0 <= epoch <= MAX_SOURCE_DATE_EPOCH)
    ):
        raise ArchiveError("archive timestamp is outside the supported unsigned 32-bit range")
    package_root = package_root.resolve()
    destination = destination.resolve()
    try:
        destination.relative_to(package_root)
    except ValueError:
        pass
    else:
        raise ArchiveError("archive destination must be outside the package root")
    if not destination.parent.is_dir():
        raise ArchiveError("archive destination directory does not exist")

    entries = source_entries(package_root)
    tar_gz = destination.name.endswith(".tar.gz")
    zip_archive = destination.suffix == ".zip"
    if not (tar_gz or zip_archive):
        raise ArchiveError("archive destination must end in .tar.gz or .zip")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        if tar_gz:
            create_tar_gz(package_root, temporary, entries, epoch)
        else:
            create_zip(package_root, temporary, entries, epoch)
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    parser.add_argument(
        "--source-date-epoch",
        default=os.environ.get("SOURCE_DATE_EPOCH"),
        help="Unsigned Unix timestamp used for every archive entry.",
    )
    args = parser.parse_args()
    if args.source_date_epoch is None:
        parser.error("--source-date-epoch or SOURCE_DATE_EPOCH is required")
    try:
        epoch = parse_source_date_epoch(args.source_date_epoch)
        create_archive(args.package_root, args.destination, epoch)
    except (ArchiveError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: created deterministic archive {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
