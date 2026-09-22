#!/usr/bin/env bash
# One-time: create the private GitHub repo and wire it as origin.
# Usage: scripts/setup_github.sh
set -euo pipefail

cd "$(dirname "$0")/.."
gh repo create laya-system-use --private --source . --remote origin
