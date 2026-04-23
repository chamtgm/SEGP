:exclamation: All codes here were written by members in SEGP Group 5 only.
# 🌟 SEGP Project Navigator

Welcome to the SEGP workspace 👋

This page is the high-level guide to help you choose the right folder quickly.
Detailed setup and execution steps are intentionally kept inside each folder-specific README.

## 🎯 Choose Your Goal

Tick the one that matches what you want to do now:

- [ ] 🖥️ Build or edit the web interface
- [ ] 🎯 Work on fruit object detection datasets or detector training
- [ ] 🍎 Work on contrastive learning, embeddings, or model evaluation
- [ ] 🔌 Work on API/model-serving logic
- [ ] 📥 Get the project dataset from Kaggle

Then use the map below 👇

## 🧭 Quick Folder Map

| If your goal is... | Go to folder | Open this guide next |
|---|---|---|
| 🖥️ Web interface and frontend experience | [webapp](webapp) | [webapp/README.md](webapp/README.md) |
| 🎯 Detection labels, YOLO dataset flow, detector outputs | [Object Detection](Object%20Detection) | [Object Detection/README.md](Object%20Detection/README.md) |
| 🍎 Self-supervised learning and classifier quality | [contrastive-fruits](contrastive-fruits) | [contrastive-fruits/README.md](contrastive-fruits/README.md) |
| 🔌 Inference service and API behavior | [scripts](scripts) | [scripts/README.md](scripts/README.md) |
| 📥 Dataset source and download | [SEGP Dataset (Kaggle)](https://www.kaggle.com/datasets/chamtgm/segp-dataset) | Use folder guides above for expected placement by workflow |

## 📥 Dataset Category

Project dataset source:

- Get your dataset here -> [SEGP Dataset on Kaggle](https://www.kaggle.com/datasets/chamtgm/segp-dataset)

Why this matters:

- 🎯 Object Detection workflows depend on dataset images for annotation and YOLO dataset generation.
- 🍎 contrastive-fruits workflows depend on dataset images for training and evaluation pipelines.
- 🔌 scripts/service workflows use model artifacts produced from those datasets.

After downloading, follow the folder-specific README for structure and expected dataset locations.

## 📦 Folder Spotlights (Expandable)

<details>
<summary><strong>🖥️ webapp</strong> - User-facing web experience</summary>

What it contains:
- 🎨 Browser UI pages and styling
- 🧩 Frontend assets
- 🔗 Node/TypeScript API gateway components

What it is responsible for:
- 📊 Presenting analysis results to users
- 🌉 Connecting UI actions to backend service endpoints

Where to continue:
- 📘 [webapp/README.md](webapp/README.md)

</details>

<details>
<summary><strong>🎯 Object Detection</strong> - Detection data and YOLO workflow</summary>

What it contains:
- 🏷️ Auto-annotation helpers
- 🗂️ Dataset preparation scripts
- 📈 YOLO dataset assets and training outputs

What it is responsible for:
- 🛠️ Producing and maintaining object detection datasets
- 🧪 Training and tracking detector artifacts

Where to continue:
- 📘 [Object Detection/README.md](Object%20Detection/README.md)

</details>

<details>
<summary><strong>🍎 contrastive-fruits</strong> - Representation learning pipeline</summary>

What it contains:
- 🧠 Contrastive learning modules and model components
- 📜 Training/evaluation scripts
- 💾 Checkpoints and analysis utilities

What it is responsible for:
- 🧬 Learning robust fruit representations
- 🧭 Supporting probing, fine-tuning, and embedding workflows

Where to continue:
- 📘 [contrastive-fruits/README.md](contrastive-fruits/README.md)

</details>

<details>
<summary><strong>🔌 scripts</strong> - Model service layer</summary>

What it contains:
- 🧾 API-oriented Python service script(s)
- ⚙️ Inference and response-shaping logic

What it is responsible for:
- 🚀 Serving model predictions and analysis endpoints
- 🌉 Bridging model artifacts with client requests

Where to continue:
- 📘 [scripts/README.md](scripts/README.md)

</details>

## ✅ One Rule For This Project

Use this README for navigation.

For any implementation, configuration, or execution detail, always follow the README inside the specific folder:
- 🖥️ [webapp/README.md](webapp/README.md)
- 🎯 [Object Detection/README.md](Object%20Detection/README.md)
- 🍎 [contrastive-fruits/README.md](contrastive-fruits/README.md)
- 🔌 [scripts/README.md](scripts/README.md)
