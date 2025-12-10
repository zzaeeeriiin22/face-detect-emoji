#!/bin/bash

emcc main.cpp geometry.cpp predict.cpp -o expression_recognition.js \
  -s EXPORTED_FUNCTIONS='["_process_frame", "_malloc", "_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap", "HEAPU8", "HEAPF32"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME='createExpressionRecognitionModule' \
  -O3 \
  -flto \
  -ffast-math \
  -msimd128 \
  -msse4.2 \
  -fno-exceptions \
  -fno-rtti \
  -DNDEBUG

if [ $? -ne 0 ]; then
    echo "Failed to build expression_recognition.js"
    exit 1
fi

echo "expression_recognition.js built successfully"