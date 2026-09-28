import sys
from pathlib import Path

# Mirrors the sys.path.append(... parent.parent) hack every script under
# scripts/ uses to resolve `src` as a top-level package -- pytest's default
# rootdir insertion doesn't add the repo root, only tests/ itself.
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
