"""Download Mind2Web train shards and the password-protected, evaluation-only test zip."""

import argparse
import logging
import zipfile
from collections.abc import Iterable
from pathlib import Path

REPO = "osunlp/Mind2Web"
TEST_ZIP_PASSWORD = b"mind2web"  # published in the dataset README; it only deters crawlers
TEST_SPLITS = ("test_task", "test_website", "test_domain")
log = logging.getLogger("fetch_data")


def fetch_train(dest: Path, shards: Iterable[int]) -> list[Path]:
    from huggingface_hub import hf_hub_download

    return [
        Path(hf_hub_download(REPO, f"data/train/train_{i}.json", repo_type="dataset", local_dir=dest))
        for i in shards
    ]


def fetch_test(dest: Path) -> dict[str, list[Path]]:
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(REPO, "test.zip", repo_type="dataset", local_dir=dest)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest / "test", pwd=TEST_ZIP_PASSWORD)
    found = {name: sorted((dest / "test").rglob(f"{name}_*.json")) for name in TEST_SPLITS}
    missing = [name for name, files in found.items() if not files]
    if missing:
        raise FileNotFoundError(f"test.zip had no files for {missing}; layout changed?")
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["train", "test"])
    parser.add_argument("--dest", type=Path, default=Path("data/mind2web"))
    parser.add_argument("--shards", type=int, nargs="+", default=list(range(11)))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.what == "train":
        for path in fetch_train(args.dest, args.shards):
            log.info("train shard: %s", path)
    else:
        for name, files in fetch_test(args.dest).items():
            log.info("%s: %d files", name, len(files))


if __name__ == "__main__":
    main()
