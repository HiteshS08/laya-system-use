#!/usr/bin/env bash
# Stage everything and commit with the given message, then push. No co-author trailer.
# Usage: scripts/commit.sh "<message>"
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: scripts/commit.sh <message>" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
git add -A
git commit -m "$1"
git push -u origin "$(git branch --show-current)"
