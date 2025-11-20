#!/bin/bash
# Initialize Ollama and download default embedding model

echo "[INFO] Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

echo "[INFO] Waiting for Ollama to be ready..."
sleep 10

# Check if Ollama is responding
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[INFO] Ollama is ready!"
        break
    fi
    echo "[INFO] Waiting for Ollama... ($i/30)"
    sleep 2
done

# Pull default embedding model
echo "[INFO] Downloading default embedding model: nomic-embed-text"
ollama pull nomic-embed-text

if [ $? -eq 0 ]; then
    echo "[INFO] Successfully downloaded nomic-embed-text model"
else
    echo "[WARNING] Failed to download nomic-embed-text model"
fi

# List available models
echo "[INFO] Available models:"
ollama list

# Keep Ollama running
wait $OLLAMA_PID

