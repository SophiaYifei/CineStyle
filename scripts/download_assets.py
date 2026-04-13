# AI tools used: Claude (Anthropic) assisted with huggingface_hub snapshot
# download integration for Railway deployment.
"""
download_assets.py — Download model weights and data from HuggingFace.

Run before starting the server in production to pull assets that are
too large for Git (crops, FAISS index, NCF/SASRec weights).
"""

import tarfile
from pathlib import Path

HF_REPO = "YifeiGuo/cinestyle-assets"


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract *tar* into *dest*, rejecting members that would escape the directory."""
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not member_path.is_relative_to(dest):
            raise ValueError(
                f"Unsafe tar member blocked (path traversal attempt): {member.name!r}"
            )
        tar.extract(member, path=dest)


def download():
    from huggingface_hub import snapshot_download

    print(f"Downloading assets from {HF_REPO} ...")
    cache = snapshot_download(repo_id=HF_REPO, repo_type="dataset")
    cache = Path(cache)

    # Symlink model directories to expected locations
    mappings = [
        (cache / "models" / "faiss_index", Path("models/faiss_index")),
        (cache / "models" / "ncf_reranker", Path("models/ncf_reranker")),
        (cache / "models" / "sasrec", Path("models/sasrec")),
    ]

    for src, dst in mappings:
        if not src.exists():
            print(f"  SKIP {src} (not found in HF repo)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            print(f"  EXISTS {dst}")
            continue
        dst.symlink_to(src)
        print(f"  LINKED {dst} -> {src}")

    # catalog.jsonl (single file)
    cat_src = cache / "catalog.jsonl"
    cat_dst = Path("data/processed/catalog.jsonl")
    if cat_src.exists() and not cat_dst.exists():
        cat_dst.parent.mkdir(parents=True, exist_ok=True)
        cat_dst.symlink_to(cat_src)
        print(f"  LINKED {cat_dst} -> {cat_src}")

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
