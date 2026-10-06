"""Read-only verification of a complete 64-rank production checkpoint."""

from pathlib import Path
import hashlib
import json
import pickle
import re


def verify_complete(path):
    """Verify all manifest sizes/metadata hashes, model headers and full states."""
    from safetensors import safe_open

    path = Path(path)
    assert (
        path.is_dir() and not path.is_symlink() and not (path / ".incomplete").exists()
    )
    assert re.fullmatch(r"epoch_\d+_step_\d+", path.name), path
    marker = path / "COMPLETE.json"
    marker_bytes = marker.read_bytes()
    complete = json.loads(marker_bytes)
    assert complete["world_size"] == 64 and complete["nodes"] == 8
    assert set(complete["node_manifests"]) == set(map(str, range(8)))
    unique = {}
    for manifest in complete["node_manifests"].values():
        assert manifest
        for relative, info in manifest.items():
            rel = Path(relative)
            assert not rel.is_absolute() and ".." not in rel.parts
            file = path / rel
            assert (
                file.resolve(strict=True).is_relative_to(path.resolve())
                and not file.is_symlink()
            )
            assert (
                file.is_file()
                and file.stat().st_size == info["bytes"]
                and info["bytes"] > 0
            )
            if info.get("sha256"):
                assert hashlib.sha256(file.read_bytes()).hexdigest() == info["sha256"]
            if relative in unique:
                assert unique[relative] == info
            else:
                unique[relative] = info
    components = {}
    for name, pattern in [
        ("model", "*.safetensors"),
        ("optim", "*.distcp"),
        ("rng", "*.pt"),
        ("dataloader", "*.pt"),
    ]:
        files = list((path / name).glob(pattern))
        assert len(files) == 64 and all(
            str(f.relative_to(path)) in unique for f in files
        ), name
        components[name] = len(files)
    for file in (path / "model").glob("*.safetensors"):
        with safe_open(file, framework="pt", device="cpu") as handle:
            assert list(handle.keys()), file
    for name in ("config.yaml", "losses.json", "step_scheduler.pt", "optim/.metadata"):
        assert name in unique, name
    # This metadata is generated locally by our training recipe, never downloaded.
    with (path / "optim/.metadata").open("rb") as stream:
        metadata = pickle.load(stream)
    assert metadata.storage_data
    for storage in metadata.storage_data.values():
        rel = Path(storage.relative_path)
        assert not rel.is_absolute() and ".." not in rel.parts
        file = path / "optim" / rel
        assert file.is_file() and file.stat().st_size >= storage.offset + storage.length
    return {
        "path": str(path),
        "complete_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        "manifest_files": len(unique),
        "manifest_bytes": sum(v["bytes"] for v in unique.values()),
        "components": components,
        "all_manifest_sizes_and_metadata_hashes_verified": True,
        "model_headers_verified": 64,
        "optimizer_storage_ranges_verified": True,
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(verify_complete(Path(sys.argv[1]).resolve()), indent=2))
