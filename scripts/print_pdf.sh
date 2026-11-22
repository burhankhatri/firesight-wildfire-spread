#!/usr/bin/env bash
# Print an HTML file to PDF with headless Chrome.  Usage: scripts/print_pdf.sh in.html out.pdf
set -euo pipefail
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
in="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --run-all-compositor-stages-before-draw \
  --virtual-time-budget=8000 --print-to-pdf="$2" "file://$in" 2>/dev/null
echo "wrote $2"
