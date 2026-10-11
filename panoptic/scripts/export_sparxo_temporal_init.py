"""Export the exact SPARXO-v2 temporal subnetwork for PaPs fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
    prefixes = ('s2_stem.', 'temporal_encoder.', 'adapter.')
    state = {
        'temp_encoder.' + key: value
        for key, value in checkpoint['model'].items()
        if key.startswith(prefixes)
    }
    if not state:
        raise RuntimeError('No SPARXO temporal tensors found in source checkpoint')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state, 'source': str(args.source)}, args.output)
    print(f'[SPARXO PaPs] exported {len(state)} temporal tensors to {args.output}')


if __name__ == '__main__':
    main()
