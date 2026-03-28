# AyurSLM - Bharat's Small Language Model for Ayurveda

AyurSLM is an open-source, lightweight Small Language Model (SLM) tailored for Ayurvedic health consultations. It is designed to understand Ayurvedic concepts (Doshas, Dinacharya, etc.) and Indian languages, allowing for culturally-aware, natural health suggestions.

## Philosophy

- **Lightweight & Accessible**: Designed to be fine-tuned and run on consumer hardware (GPUs or even CPUs with quantization).
- **Culturally Relevant**: Trained on Ayurvedic principles, prioritizing Vata, Pitta, Kapha balancing.
- **Multilingual**: Capable of understanding and responding in Bharatiya languages (e.g., Hindi, Tamil, Telugu) by expanding the dataset.
- **Strictly Advisory**: Configured to avoid making modern medical diagnoses, focusing instead on lifestyle (Vihar) and dietary (Aahar) natural remedies.

## Project Structure

1. `train.py`: Script to fine-tune a fast, lightweight base model (e.g., Llama 3.2 1B) on an Ayurvedic dataset. It uses `unsloth` for 2x faster training and significantly lower memory usage.
2. `app.py`: A Gradio web interface to interact with your locally trained AyurSLM.
3. `sample_dataset.jsonl`: A starter dataset demonstrating the instruction-response format for Ayurvedic consultations, including an example in Hindi.

## Setup Instructions

It is recommended to run training on a machine with an NVIDIA GPU. Windows users can also use WSL2.

1. Create a Python virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install the necessary deep learning libraries and Unsloth:
```powershell
pip install -r requirements.txt
```
*(Note for Windows users: If you run into issues installing `unsloth`, please visit the [Unsloth GitHub](https://github.com/unslothai/unsloth) for the exact Pip command corresponding to your CUDA version).*

## How to Train

1. Expand `sample_dataset.jsonl` with your specific dataset. You can incorporate existing HuggingFace datasets or use translated texts from classical literature.
2. Run the fine-tuning script:
```powershell
python train.py
```
This script will load the base model, train the LoRA adapters using the dataset, and save the finalized adapter to the `ayurslm-lora` folder.

## How to Run the Inference App

Once the model has finished training, you can launch the conversational UI:
```powershell
python app.py
```
Open the provided local URL (usually `http://127.0.0.1:7860`) in your browser. You can describe your symptoms in English or Hindi, and AyurSLM will suggest which Dosha might be out of balance along with dietary and lifestyle recommendations!
