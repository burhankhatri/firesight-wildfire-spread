#!/usr/bin/env bash
# Assemble the static website in site/dist (results, figures, twin data, PDFs).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf site/dist && mkdir -p site/dist/figures && cp -R site/.vercel site/dist/
cp site/index.html site/dist/
cp outputs/figures/*.png site/dist/figures/
cp outputs/results.json outputs/twin.json site/dist/
[ -f paper/FireSight_Paper.pdf ] && cp paper/FireSight_Paper.pdf site/dist/
[ -f slides/FireSight_Slides.pdf ] && cp slides/FireSight_Slides.pdf site/dist/
echo "site/dist ready"
