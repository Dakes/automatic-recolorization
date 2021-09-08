#!/usr/bin/env bash
set -euo pipefail

cd colorization-pytorch
chmod +x ./pretrained_models/download_siggraph_model.sh
./pretrained_models/download_siggraph_model.sh
cd ..

cd interactive-deep-colorization
chmod +x ./models/fetch_models.sh
./models/fetch_models.sh
cd ..
