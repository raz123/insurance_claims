# Project: Privacy-First Local OCR Claims System 🚀

## 🎯 Objective
To build a 100% client-side, zero-cost insurance claims application that performs high-fidelity OCR entirely within the browser using WebGPU. No external APIs, no cloud processing, no recurring costs.

## 🧠 Core Engine: GLM-OCR
- **Performance**: Ranked #1 overall on OmniDocBench V1.5 (94.62 score).
- **Architecture**: CogViT Vision Encoder + Llama-based GLM-0.5B Decoder.
- **Implementation**: Custom ONNX Runtime Web orchestration (`onnxruntime-web` + WebGPU).
- **Spatial Awareness**: Manual 3D-position ID generation for document layout mapping.

## 🛠️ Technology Stack
- **Frontend**: Vanilla HTML5/CSS3 (Glassmorphism UI).
- **Inference**: WebWorkers + ONNX Runtime (WebGPU backend).
- **Persistence**: Google Apps Script (DB Save to Google Sheets).
- **Privacy**: Local-only inference. Document data never leaves the user's browser.

## 📈 Benchmarks
- **Lightweight**: 0.9B parameters (approx. 1.2GB VRAM).
- **Speed**: Optimized for <5s inference on modern integrated GPUs.
- **Accuracy**: State-of-the-art on complex tables, handwritten receipts, and mixed-language documents.

## 🧑‍💻 Handover Checklist
- [x] WebWorker-based ONNX pipeline.
- [x] 3D Position ID mapping (Temporal/Sequential/Row/Col).
- [x] 2x2 Spatial Token Merging.
- [x] Structured JSON Schema Prompting.
- [x] Google Sheets Backend Integration.
