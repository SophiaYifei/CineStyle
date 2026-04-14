# AI tools used: Claude (Anthropic) assisted with huggingface_hub snapshot
# download integration for Railway deployment.
"""
download_assets.py — Download model weights and data from HuggingFace.

Run before starting the server in production to pull assets that are
too large for Git (crops, FAISS index, NCF/SASRec weights).
"""

import shutil
import tarfile
from pathlib import Path

HF_REPO = "YifeiGuo/cinestyle-assets"


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract tar into dest, rejecting path-traversal members."""
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not member_path.is_relative_to(dest):
            raise ValueError(f"Unsafe tar member: {member.name!r}")
        tar.extract(member, path=dest)


def _copy_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy all files from src_dir into dst_dir (creating dst_dir if needed)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.iterdir():
        if src_file.is_file():
            dst_file = dst_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"  COPIED {src_file.name} -> {dst_file}")
            else:
                print(f"  EXISTS {dst_file}")


def download():
    from huggingface_hub import snapshot_download

    print(f"Downloading assets from {HF_REPO} ...")
    cache = Path(snapshot_download(repo_id=HF_REPO, repo_type="dataset"))

    # Copy model files individually into expected locations
    mappings = [
        (cache / "models" / "faiss_index", Path("models/faiss_index")),
        (cache / "models" / "ncf_reranker", Path("models/ncf_reranker")),
        (cache / "models" / "sasrec", Path("models/sasrec")),
    ]
    for src, dst in mappings:
        if not src.exists():
            print(f"  SKIP {src} (not in HF repo)")
            continue
        _copy_files(src, dst)

    # catalog.jsonl (single file)
    cat_src = cache / "catalog.jsonl"
    cat_dst = Path("data/processed/catalog.jsonl")
    if cat_src.exists() and not cat_dst.exists():
        cat_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cat_src, cat_dst)
        print(f"  COPIED catalog.jsonl -> {cat_dst}")

    # Extract crops from tar.gz archive
    crops_dst = Path("data/raw/crops")
    crops_tar = cache / "crops.tar.gz"
    if crops_tar.exists() and not crops_dst.exists():
        print(f"  Extracting {crops_tar} ...")
        extract_dir = Path("data/raw").resolve()
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(crops_tar, "r:gz") as tar:
            _safe_extractall(tar, extract_dir)
        print(f"  EXTRACTED crops -> {crops_dst}")
    elif crops_dst.exists():
        print(f"  EXISTS {crops_dst}")

    print("Asset download complete.")


if __name__ == "__main__":
    download()
