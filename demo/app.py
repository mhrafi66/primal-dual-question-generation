#!/usr/bin/env python3
"""Small Gradio demo for a trained answer-aware seq2seq checkpoint.

Run:
    QG_MODEL=outputs/bart-answer-aware python demo/app.py
"""
from __future__ import annotations

import os

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = os.environ.get("QG_MODEL", "outputs/bart-answer-aware")

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(_device).eval()
    return _tokenizer, _model


def generate(context: str, answer: str, beams: int) -> str:
    if not context.strip() or not answer.strip():
        return "Please provide both a passage and a target answer."
    tokenizer, model = load_model()
    prompt = f"answer: {answer.strip()} context: {context.strip()}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(_device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            num_beams=int(beams),
            max_new_tokens=48,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=8, label="Passage"),
        gr.Textbox(lines=2, label="Target answer"),
        gr.Slider(1, 8, value=4, step=1, label="Beam width"),
    ],
    outputs=gr.Textbox(label="Generated question"),
    title="Answer-Aware Question Generation",
    description=(
        "Portfolio demo for the answer-aware question generation project. "
        "Set QG_MODEL to a local fine-tuned seq2seq checkpoint before launching."
    ),
)

if __name__ == "__main__":
    demo.launch()
