# Customer Care AI Model

## Prerequisites & Local Model Setup (Ollama)

This project runs completely locally using [Ollama](https://ollama.com) on `http://localhost:11434`.

### 1. Install Ollama (By OS)

#### Windows
1. Download the official Windows installer from [://ollama.com](https://ollama.comdownload/windows) (or install via `winget`):
   ```powershell
   winget install Ollama.Ollama
   ```
2. Run `OllamaSetup.exe` and follow the on-screen prompts.
3. Once installed, Ollama starts automatically in the background (check the Ollama icon in the Windows system tray).

#### Linux
1. Run the official install script in your terminal:
   ```bash
   curl -fsSL https://ollama.cominstall.sh | sh
   ```
2. Start and enable the Ollama background service (if not started automatically):
   ```bash
   sudo systemctl enable --now ollama
   ```
   *(Alternatively, run `ollama serve` manually in a separate terminal window).*

---

### 2. Download Required Models Locally

Open your terminal (PowerShell on Windows or Bash on Linux) and pull the embedding and chat models used in this project:

```bash
# 1. Pull the embedding model
ollama pull nomic-embed-text

# 2. Pull the chat model
ollama pull qwen3:4b-instruct
```

---

### 3. Verify Ollama & Models

1. Confirm both models are installed locally:
   ```bash
   ollama list
   ```
   You should see both `nomic-embed-text` and `qwen3:4b-instruct` in the output.

2. Verify the local API endpoint (`http://localhost:11434`) is active:
   ```bash
   curl http://localhost:11434
   ```
   Expected output: `Ollama is running`

---

### 4. Run the Application

With Ollama running in the background and your markdown documents placed inside the `docs/` folder, start the RAG script using `uv`:

```bash
uv run rag.py
```
