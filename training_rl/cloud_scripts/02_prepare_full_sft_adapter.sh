#!/usr/bin/env bash
set -euo pipefail
ROOT=/hy-tmp/rema_mappo_v0_tiny_20260615
TAR=${FULL_SFT_TAR:-/hy-tmp/rema_full_sft_v4_results_20260614_185318.tar.gz}
DST="$ROOT/adapters/full_sft_v4"
mkdir -p "$DST"
test -s "$TAR"
tmp="$ROOT/_full_sft_extract"
rm -rf "$tmp"
mkdir -p "$tmp"
tar -xzf "$TAR" -C "$tmp" LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/full_sft_v4
cp -a "$tmp"/LLaMA-Factory/saves/llama3-8b/lora/rema_plantuml/full_sft_v4/. "$DST"/
test -s "$DST/adapter_model.safetensors"
test -s "$DST/adapter_config.json"
echo "Prepared Full-SFT adapter at $DST"
