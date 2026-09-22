"""Restore the exact evidence files from the lossless HDF5 delivery container."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import h5py


def extract(source, destination):
    if destination.exists():
        raise ValueError('Destination already exists; choose an empty new path. No files are overwritten.')
    with h5py.File(source, 'r') as h:
        if h.attrs.get('format') != 'paper4-evidence-files-v1':
            raise ValueError('Not a supported Paper 4 evidence container')
        manifest_bytes = h['manifest_json'][...].tobytes()
        manifest = json.loads(manifest_bytes)
        seen = set()
        total = 0
        for row in manifest['files']:
            p = PurePosixPath(row['path'])
            if p.is_absolute() or '..' in p.parts or '\\' in row['path'] or p.as_posix() in seen:
                raise ValueError('Unsafe or repeated file path: '+row['path'])
            seen.add(p.as_posix())
            total += row['bytes']
            if row['bytes'] != h['objects'][row['sha256']].size:
                raise ValueError('Stored byte count mismatch: '+row['path'])
        if total > 4_000_000_000:
            raise ValueError('Unexpected payload size for this bounded dataset')
        destination.mkdir(parents=True)
        for row in manifest['files']:
            payload = h['objects'][row['sha256']][...].tobytes()
            if hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError('Checksum mismatch: '+row['path'])
            path = destination/row['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(payload)
        with (destination/'MANIFEST.json').open('xb') as stream:
            stream.write(manifest_bytes)
    print(f'Restored and SHA-256 checked {len(seen)} files ({total:,} bytes).')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('container', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    extract(a.container, a.output)
