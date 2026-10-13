"""Training core for the LightOnOCR LoRA: prompt, pixel budget, LoRA targets, collator, generation.

This subpackage needs .venv-lighton (torch, transformers, peft). It is never imported by
lightonocr_math/__init__.py, so `import lightonocr_math` stays torch-free in the project venv.
"""
