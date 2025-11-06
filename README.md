# Project Ranni

A comprehensive data pipeline for collecting and managing gameplay data from videos and live gameplay sessions.

## Overview

This project consists of two main pipelines:

1. **YouTube Data Pipeline** (`yt_data_pipeline`) - Download, process, and extract data from gameplay videos
2. **Gameplay Pipeline** (`gameplay_pipeline`) - Record live gameplay and upload to Hugging Face

## Installation

This project uses `uv` for dependency management. Install dependencies:

```bash
# Install all dependencies
uv sync

# Or install specific pipeline dependencies
uv sync --group yt_data_pipeline
uv sync --group gameplay_pipeline
```

## YouTube Data Pipeline

Process gameplay videos from YouTube or local sources.

### Features

- Download videos from YouTube using `yt-dlp`
- Extract frames from videos
- Detect game UI elements (HP, stamina, etc.) using OCR
- Trim videos to specific segments
- Export processed data to pickle/HDF5 format

### Usage

```bash
# Run the full pipeline
uv run -m yt_data_pipeline
```

Individual steps can be run separately - see the scripts in `yt_data_pipeline/`.

## Gameplay Pipeline

Record live gameplay using the [pysiphon](https://github.com/dhmnr/pysiphon) library and upload to Hugging Face.

### Prerequisites

1. **Siphon Server**: You need a running Siphon server that provides:
   - Memory attribute access (health, position, etc.)
   - Screen capture capabilities
   - Input control (optional)

2. **Configuration**: Create a `siphon_config.toml` file with your game's memory addresses and attributes. See `gameplay_pipeline/siphon_config.toml.example` for a template.

### Features

- Connect to Siphon server for game data access
- Record gameplay sessions with:
  - Video frames (screen capture)
  - Game attributes (health, mana, position, etc.)
  - Synchronized HDF5 output
- Live preview of recording status
- Automatic upload to Hugging Face datasets

### Usage

#### Test Connection and Capture

```bash
# Test that you can capture frames
uv run -m gameplay_pipeline.record_gameplay \
    --host localhost:50051 \
    --config siphon_config.toml \
    --attributes health,mana \
    --test-capture
```

#### Record a Session

```bash
# Record 60 seconds of gameplay
uv run -m gameplay_pipeline.record_gameplay \
    --host localhost:50051 \
    --config siphon_config.toml \
    --attributes health,mana,stamina,position \
    --duration 60 \
    --output-dir ./recordings
```

#### Record and Upload to Hugging Face

```bash
# Record and automatically upload to HF
uv run -m gameplay_pipeline.record_gameplay \
    --host localhost:50051 \
    --config siphon_config.toml \
    --attributes health,mana,stamina,position \
    --duration 300 \
    --upload \
    --repo-id username/my-gameplay-dataset \
    --hf-token hf_xxx
```

### Command-Line Options

- `--host` - Siphon server address (default: `localhost:50051`)
- `--config` - Path to Siphon configuration file
- `--attributes` - Comma-separated list of attributes to record (required)
- `--output-dir` - Output directory for recordings (default: `./recordings`)
- `--duration` - Maximum recording duration in seconds (default: 60)
- `--no-preview` - Disable live attribute preview
- `--test-capture` - Test frame capture and exit
- `--upload` - Upload recording to Hugging Face after completion
- `--repo-id` - Hugging Face repository ID (e.g., `username/repo-name`)
- `--hf-token` - Hugging Face API token

### Programmatic Usage

```python
from gameplay_pipeline.record_gameplay import GameplayRecorder
from gameplay_pipeline.hf_upload import upload_to_huggingface

# Record gameplay
with GameplayRecorder(
    host="localhost:50051",
    config_path="siphon_config.toml"
) as recorder:
    # Test capture
    recorder.test_capture("test.png")
    
    # Get current attribute values
    attrs = recorder.get_attributes(["health", "mana"])
    print(f"Current health: {attrs['health']}")
    
    # Record a session
    result = recorder.record_session(
        attribute_names=["health", "mana", "stamina"],
        output_directory="./recordings",
        max_duration_seconds=120,
    )
    
    print(f"Recording saved to: {result['file_path']}")
    print(f"Total frames: {result['stats']['total_frames']}")

# Upload to Hugging Face
upload_to_huggingface(
    file_path=result['file_path'],
    repo_id="username/my-dataset",
    token="hf_xxx"
)
```

## Hugging Face Upload

Upload any file or folder to Hugging Face Hub:

```bash
# Upload a single file
uv run -m gameplay_pipeline.hf_upload my_recording.h5 username/repo-name

# Or use programmatically
from gameplay_pipeline.hf_upload import upload_to_huggingface, upload_folder_to_huggingface

# Upload file
upload_to_huggingface(
    file_path="recording.h5",
    repo_id="username/my-dataset",
    repo_type="dataset",
    token="hf_xxx"
)

# Upload entire folder
upload_folder_to_huggingface(
    folder_path="./recordings",
    repo_id="username/my-dataset",
    path_in_repo="recordings/session1",
    ignore_patterns=["*.tmp", "__pycache__"]
)
```

## Project Structure

```
ProjectRanni/
├── yt_data_pipeline/          # YouTube video processing
│   ├── download_videos.py     # Download from YouTube
│   ├── extract_frames.py      # Extract video frames
│   ├── hp_detection.py        # Detect UI elements
│   ├── trim_videos.py         # Trim videos
│   └── video_to_pickle.py     # Export to pickle/HDF5
├── gameplay_pipeline/         # Live gameplay recording
│   ├── record_gameplay.py     # Main recording script
│   ├── hf_upload.py          # Hugging Face upload utilities
│   └── siphon_config.toml.example  # Example configuration
├── data/                      # Data storage
│   └── videos/               # Downloaded and processed videos
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## Authentication

### Hugging Face

Set your Hugging Face token as an environment variable:

```bash
# Windows PowerShell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxx"

# Windows CMD
set HF_TOKEN=hf_xxxxxxxxxxxxx

# Linux/Mac
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

Or pass it directly via `--hf-token` argument.

## Output Format

Recordings are saved in HDF5 format (`.h5` files) with the following structure:

- **frames**: Video frames captured during gameplay
- **attributes**: Time-series data for each recorded attribute
- **metadata**: Recording information (FPS, duration, etc.)

You can read HDF5 files using:

```python
import h5py

with h5py.File("recording.h5", "r") as f:
    frames = f["frames"][:]
    health = f["attributes/health"][:]
    metadata = dict(f.attrs)
```

## Development

This project uses:
- **uv** for dependency management
- **Python 3.10+** for type hints and modern features
- **pysiphon** for gameplay data access
- **huggingface-hub** for dataset uploads
- **rich** for beautiful CLI output

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- All features are documented
- Dependencies are properly declared in `pyproject.toml`

