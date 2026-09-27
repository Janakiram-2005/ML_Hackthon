# ML Hackathon Project

Data is strictly located locally in `./data/`.

This repository contains the model architecture, training orchestrator, and FastAPI serving backend.

## Run tests
```bash
pytest tests/
```

## Train Model
```bash
python src/train.py
```

## Run API
```bash
uvicorn app.main:app --reload
```
