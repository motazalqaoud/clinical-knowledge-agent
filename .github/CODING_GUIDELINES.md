# Coding Guidelines - Motaz Alqaoud

## Project: Clinical Knowledge Agent - RAG system for clinical documents

## About
Healthcare AI project by Motaz Alqaoud, PhD - Senior AI/ML Engineer.
Part of the Precision Oncology AI Platform (9-module portfolio).

## Coding style
- Python 3.11+, type hints on all functions
- Google-style docstrings with Args, Returns, and Clinical note section
- Black formatting (line length 88)

## Clinical AI rules
- Never hardcode drug dosages without a physician-confirmation note
- Always include safety disclaimers in patient-facing output
- Prefer explicit error messages over silent failures

## Preferred libraries
- PyTorch, MONAI, sentence-transformers, FAISS, Gradio, pytest
