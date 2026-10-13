"""The one model, resident on the AMD GPU: transcribes images during --index
and answers questions afterwards.

ROCm builds of PyTorch expose the AMD GPU through the `torch.cuda` API (HIP
underneath), so "cuda" here means the MI300X / Navi 31 card. We refuse to run
on the CPU: a submission whose VRAM never clears 1 GiB is rejected.

The weights are read from MC3_VLM_PATH (default /models/vlm), shipped in the
image. Offline mode is forced so a missing file fails fast instead of hanging
on a network that does not exist at evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

VLM_PATH = os.environ.get("MC3_VLM_PATH", "/models/vlm")
MAX_SIDE = 1600
OCR_MAX_NEW_TOKENS = 512

OCR_PROMPT = (
    "Transcribe ALL text in this image exactly as written, line by line. "
    "Preserve label/value pairs on one line (e.g. 'BOARD REVISION: REV-C2'). "
    "If it is a diagram, list every box, pin or component together with the "
    "label connected to it (e.g. 'B14: THERM_ALERT#'). "
    "Output plain text only. Do not describe or interpret the image."
)


class Model:
    def __init__(self, path: str = VLM_PATH):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError(
                "no GPU visible to PyTorch (ROCm). Refusing to run on the CPU -- "
                "check --device /dev/kfd --device /dev/dri and HIP_VISIBLE_DEVICES")
        if not Path(path).is_dir():
            raise FileNotFoundError(f"model weights not found at {path}; ship them in the image")

        self.torch = torch
        self.device = torch.device("cuda:0")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True,
        ).to(self.device).eval()

    def chat(self, system: str, user: str, max_new_tokens: int = 96, prefill: str = "") -> str:
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "text", "text": user}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True) + prefill
        inputs = self.processor(text=[prompt], return_tensors="pt").to(self.device)
        return self._generate(inputs, max_new_tokens)

    def transcribe(self, abspath: Path, job: dict) -> str:
        image = self._load(Path(abspath), job)
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": OCR_PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)
        return self._generate(inputs, OCR_MAX_NEW_TOKENS)

    def _generate(self, inputs, max_new_tokens: int) -> str:
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    @staticmethod
    def _load(abspath: Path, job: dict):
        from PIL import Image, ImageOps

        if job["kind"] == "pdf_page":
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(abspath))
            try:
                image = pdf[job["page"]].render(scale=2.0).to_pil()
            finally:
                pdf.close()
        else:
            with Image.open(abspath) as im:
                im.load()  # raises UnidentifiedImageError / OSError on a corrupt file
                image = ImageOps.exif_transpose(im)
        image = image.convert("RGB")
        if max(image.size) > MAX_SIDE:
            image.thumbnail((MAX_SIDE, MAX_SIDE))
        return image
