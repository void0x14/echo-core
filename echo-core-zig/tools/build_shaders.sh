#!/bin/sh
# GLSL -> SPIR-V; .spv dosyaları repoda: `zig build` glslc istemez.
set -e
cd "$(dirname "$0")/../src/llama/gpu/shaders"
C="glslc --target-env=vulkan1.1 -O -fshader-stage=compute"
$C -DL=64 -DW4=1 mv.comp -o mv1024.spv
$C -DL=64 -DW4=2 mv.comp -o mv2048.spv
$C -DL=56 -DW4=4 mv.comp -o mv3584.spv
for f in *.comp; do
  case "$f" in mv.comp) ;; *) $C "$f" -o "${f%.comp}.spv" ;; esac
done
