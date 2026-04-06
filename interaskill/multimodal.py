"""
Multimodal encoder and skill prediction from webpage screenshots.

Implements the paper's multimodal grounding framework (Section 3.3):
  φ_multi(v,l,a,t) = [φ_vision(v); φ_text(l); φ_action(a); φ_temporal(t)]

Components:
  1. Qwen3-VL-8B: Screenshot → skill prediction (end-to-end VLM)
  2. CLIP ViT-L/14: Screenshot → embedding for cross-domain alignment (Eq 3.8)
  3. Visual-semantic alignment loss L_align (Eq 3.8)

Usage:
    from interaskill.multimodal import VLMSkillPredictor, CLIPEncoder

    # Screenshot-based skill prediction
    vlm = VLMSkillPredictor("Qwen/Qwen3-VL-8B-Instruct")
    skill = vlm.predict_skill(screenshot, task_description, history)

    # CLIP embedding for cross-domain transfer
    clip = CLIPEncoder()
    emb = clip.encode_image(screenshot)
    sim = clip.compute_similarity(screenshot_a, screenshot_b)
"""

import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from .data import SKILL_TYPES


# ── CLIP Encoder (Eq 3.8: Visual-Semantic Alignment) ────────────────

class CLIPEncoder:
    """CLIP ViT-L/14 encoder for visual-semantic alignment and cross-domain transfer.

    Implements:
      Sim(s_A, s_B) = cos(φ_CLIP(s_A), φ_CLIP(s_B)) > θ  (Eq 3.8)
      L_align = ||φ_vision(v_region) - φ_text(l_label)||²  (Eq 3.8)
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 device: str = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP: {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print(f"  CLIP loaded on {self.device}")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a screenshot to CLIP embedding space.

        Args:
            image: PIL Image (webpage screenshot)
        Returns:
            Normalized embedding (1, d_clip)
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text (skill label, button text, etc.) to CLIP embedding space."""
        inputs = self.processor(text=text, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        emb = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1)

    def compute_similarity(self, image_a: Image.Image,
                           image_b: Image.Image) -> float:
        """Compute cosine similarity between two screenshots (Eq 3.8).

        Used for cross-domain transfer: measure visual similarity
        of equivalent UI states across different websites.
        """
        emb_a = self.encode_image(image_a)
        emb_b = self.encode_image(image_b)
        return torch.nn.functional.cosine_similarity(emb_a, emb_b).item()

    def compute_alignment_loss(self, image: Image.Image,
                               label: str) -> float:
        """Compute L_align = ||φ_vision(v) - φ_text(l)||² (Eq 3.8).

        Measures how well visual and semantic representations align.
        Lower = better alignment between what the element looks like
        and what it means semantically.
        """
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(label)
        return (img_emb - txt_emb).pow(2).sum().item()

    def batch_encode_images(self, images: list[Image.Image],
                            batch_size: int = 16) -> torch.Tensor:
        """Encode a batch of screenshots. Returns (N, d_clip)."""
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt",
                                    padding=True).to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
                emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)

    def skill_label_embeddings(self) -> dict[str, torch.Tensor]:
        """Pre-compute CLIP embeddings for all 12 skill labels.

        Useful for zero-shot skill classification from screenshots.
        """
        # Descriptive prompts for each skill (more informative than bare labels)
        skill_descriptions = {
            "search_navigate": "searching and navigating on a webpage, using search bar or clicking links",
            "document_edit": "editing a document, typing text, filling out a form",
            "review_content": "reviewing content, reading information, checking details on a page",
            "export_publish": "exporting or publishing content, downloading files, printing",
            "send_message": "sending a message, composing email, writing in chat",
            "collaborate": "collaborating with others, sharing documents, team communication",
            "schedule_meeting": "scheduling a meeting, selecting dates on calendar",
            "data_transfer": "transferring data, selecting options, configuring settings",
            "organize_files": "organizing files, managing folders, sorting items",
            "presentation_edit": "editing a presentation, adding slides, formatting",
            "monitor_status": "monitoring status, checking dashboard, viewing progress",
            "generic_action": "clicking a button, confirming an action, general interaction",
        }
        embeddings = {}
        for skill in SKILL_TYPES:
            desc = skill_descriptions.get(skill, skill.replace("_", " "))
            embeddings[skill] = self.encode_text(desc).cpu()
        return embeddings

    def zero_shot_classify(self, image: Image.Image) -> tuple[str, float]:
        """Zero-shot skill classification from screenshot using CLIP.

        Compares screenshot embedding against all skill label embeddings.
        Returns (predicted_skill, confidence).
        """
        img_emb = self.encode_image(image)
        skill_embs = self.skill_label_embeddings()

        best_skill = "generic_action"
        best_sim = -1.0
        for skill, emb in skill_embs.items():
            sim = torch.nn.functional.cosine_similarity(
                img_emb, emb.to(self.device)).item()
            if sim > best_sim:
                best_sim = sim
                best_skill = skill

        return best_skill, best_sim


# ── Qwen3-VL Skill Predictor ────────────────────────────────────────

class VLMSkillPredictor:
    """Screenshot-based skill prediction using Qwen3-VL-8B.

    Implements the full multimodal encoder:
      φ_multi(v,l,a,t) = [φ_vision(v); φ_text(l); φ_action(a); φ_temporal(t)]

    The VLM processes the screenshot (v), task description (l),
    action history (a), and temporal context (t) jointly.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
                 device_map: str = "auto"):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from transformers import BitsAndBytesConfig

        self.model_name = model_name
        print(f"Loading VLM: {model_name}...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True)

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        print(f"  VLM loaded, GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        self.skill_list = ", ".join(SKILL_TYPES)

    def predict_skill(self, screenshot: Image.Image, task: str,
                      history: str = "",
                      max_new_tokens: int = 100) -> str:
        """Predict next skill from webpage screenshot + task context.

        Args:
            screenshot: PIL Image of current webpage state
            task: Task description (e.g., "Find a red jacket under $50")
            history: Previous action history string
            max_new_tokens: Max tokens to generate

        Returns:
            Predicted skill name
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text":
                    f"You are a computer-using agent that completes web tasks by executing skills.\n"
                    f"Available skills: {self.skill_list}\n\n"
                    f"Given the screenshot of the current webpage, the task description, "
                    f"and action history, predict the NEXT skill to execute.\n"
                    f"Respond with: [Action: skill_name]"
                }],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text":
                        f"Task: {task}\n\n"
                        f"Action history: {history[-500:] if history else 'None'}\n\n"
                        f"Looking at this webpage screenshot, what skill should be executed next?"
                    },
                ],
            },
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[screenshot],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the new tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated, skip_special_tokens=True).strip()
        return self._extract_skill(response)

    def predict_action(self, screenshot: Image.Image, task: str,
                       skill: str, history: str = "",
                       max_new_tokens: int = 150) -> str:
        """Predict concrete action from screenshot + skill context.

        Two-stage: given the predicted skill, the VLM sees the screenshot
        and selects the specific element/action to execute.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text":
                        f"You are executing the skill \"{skill}\" on this webpage.\n\n"
                        f"Task: {task}\n"
                        f"Action history: {history[-500:] if history else 'None'}\n\n"
                        f"What specific action should be taken? Respond with:\n"
                        f"element_description | action_type | value\n"
                        f"Where action_type is: click, type, or select_option"
                    },
                ],
            },
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[screenshot],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def _extract_skill(self, response: str) -> str:
        """Extract skill name from VLM response."""
        match = re.search(r"\[Action:\s*(\w+)\]", response, re.IGNORECASE)
        if match:
            skill = match.group(1).lower()
            if skill in set(SKILL_TYPES):
                return skill

        response_lower = response.lower()
        for skill in SKILL_TYPES:
            if skill in response_lower:
                return skill
        return "generic_action"


# ── Screenshot Utilities ─────────────────────────────────────────────

def html_to_screenshot(html: str, width: int = 1280,
                       height: int = 800) -> Image.Image:
    """Render HTML to a screenshot image using Selenium.

    Falls back to a placeholder if Selenium/Chrome not available.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import tempfile, os

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--window-size={width},{height}")

        driver = webdriver.Chrome(options=options)

        # Write HTML to temp file
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write(html)
            tmp_path = f.name

        driver.get(f"file://{tmp_path}")
        screenshot_bytes = driver.get_screenshot_as_png()
        driver.quit()
        os.unlink(tmp_path)

        from io import BytesIO
        return Image.open(BytesIO(screenshot_bytes))

    except Exception:
        # Fallback: create a placeholder image with text
        img = Image.new("RGB", (width, height), color=(240, 240, 240))
        return img


def crop_element_region(screenshot: Image.Image,
                        bbox: dict) -> Image.Image:
    """Crop a UI element region from a screenshot.

    Args:
        screenshot: Full page screenshot
        bbox: {"x": float, "y": float, "width": float, "height": float}
              Coordinates in pixels or normalized [0,1]
    """
    w, h = screenshot.size
    x = bbox.get("x", 0)
    y = bbox.get("y", 0)
    bw = bbox.get("width", w)
    bh = bbox.get("height", h)

    # If normalized coordinates (0-1), convert to pixels
    if x <= 1.0 and y <= 1.0:
        x, y, bw, bh = x * w, y * h, bw * w, bh * h

    left = max(0, int(x))
    top = max(0, int(y))
    right = min(w, int(x + bw))
    bottom = min(h, int(y + bh))

    return screenshot.crop((left, top, right, bottom))


# ── Cross-Domain Visual Similarity ───────────────────────────────────

def compute_cross_domain_similarity(
    clip_encoder: CLIPEncoder,
    domain_screenshots: dict[str, list[Image.Image]],
) -> dict:
    """Compute pairwise visual similarity between domains (Eq 3.8).

    Args:
        clip_encoder: Loaded CLIPEncoder
        domain_screenshots: {"travel": [img1, img2, ...], "shopping": [...]}

    Returns:
        Dict with per-domain-pair similarities and statistics.
    """
    domains = sorted(domain_screenshots.keys())
    domain_embeddings = {}

    for domain in domains:
        imgs = domain_screenshots[domain]
        if imgs:
            embs = clip_encoder.batch_encode_images(imgs)
            domain_embeddings[domain] = embs.mean(dim=0, keepdim=True)

    # Pairwise similarity
    similarities = {}
    for i, d_a in enumerate(domains):
        for d_b in domains[i + 1:]:
            if d_a in domain_embeddings and d_b in domain_embeddings:
                sim = torch.nn.functional.cosine_similarity(
                    domain_embeddings[d_a],
                    domain_embeddings[d_b],
                ).item()
                similarities[f"{d_a}_vs_{d_b}"] = sim

    return {
        "pairwise_similarities": similarities,
        "mean_cross_domain_sim": float(np.mean(list(similarities.values()))) if similarities else 0.0,
        "n_domains": len(domains),
    }


def compute_skill_visual_invariance(
    clip_encoder: CLIPEncoder,
    skill_screenshots: dict[str, dict[str, list[Image.Image]]],
) -> dict:
    """Measure skill invariance across domains (Eq 3.9).

    Tests: does "search_navigate" look similar across different websites?

    Args:
        skill_screenshots: {skill: {domain: [screenshots]}}

    Returns:
        Per-skill invariance scores (lower = more invariant = better transfer).
    """
    results = {}
    for skill in sorted(skill_screenshots.keys()):
        domains = skill_screenshots[skill]
        domain_names = sorted(domains.keys())

        if len(domain_names) < 2:
            continue

        # Compute mean embedding per domain for this skill
        domain_embs = {}
        for d in domain_names:
            if domains[d]:
                embs = clip_encoder.batch_encode_images(domains[d])
                domain_embs[d] = embs.mean(dim=0)

        # Compute variance across domains (invariance measure)
        if len(domain_embs) >= 2:
            all_embs = torch.stack(list(domain_embs.values()))
            # Invariance = mean pairwise distance (lower = more invariant)
            n = all_embs.shape[0]
            pairwise_sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = torch.nn.functional.cosine_similarity(
                        all_embs[i:i+1], all_embs[j:j+1]).item()
                    pairwise_sims.append(sim)

            results[skill] = {
                "mean_cross_domain_similarity": float(np.mean(pairwise_sims)),
                "invariance": 1.0 - float(np.mean(pairwise_sims)),  # lower = better
                "n_domains": len(domain_embs),
                "n_screenshots": sum(len(v) for v in domains.values()),
            }

    return results
