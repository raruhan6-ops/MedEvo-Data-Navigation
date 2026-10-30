"""Generation backends shared by 02_eval.py and 04_rollouts.py.

  vllm          fast; Linux + CUDA only
  transformers  fallback when vLLM will not install or run
  api           OpenAI-compatible endpoint (e.g. Qwen on DashScope) for C5

All backends take the same prompt (common.PROMPT) and return, per row, a list
of n response strings.
"""
import base64
import os

from common import PROMPT, image_path, load_image


def user_text(row):
    return PROMPT.format(question=row["question"])


def chat_prompt(processor, msgs):
    # enable_thinking=False gives Qwen3.5 a direct (non-thinking) answer; templates
    # without that switch (e.g. Qwen2.5-VL) ignore it.
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def user_messages(row):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text(row)}]}]


def make_backend(name, model, max_new_tokens=256, **kw):
    if name == "auto" and kw.get("adapter"):
        name = "transformers"  # LoRA adapters (laptop training) are loaded with PEFT
    if name == "auto":
        try:
            import vllm  # noqa: F401
            name = "vllm"
        except ImportError:
            name = "transformers"
    print(f"backend: {name}  model: {model}" + (f"  adapter: {kw['adapter']}" if kw.get("adapter") else ""))
    return {"vllm": VLLMBackend, "transformers": HFBackend, "api": APIBackend}[name](
        model, max_new_tokens, **kw)


class VLLMBackend:
    def __init__(self, model, max_new_tokens, gpu_mem=0.9, max_model_len=4096, **_):
        from transformers import AutoProcessor
        from vllm import LLM

        self.processor = AutoProcessor.from_pretrained(model)
        self.llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       limit_mm_per_prompt={"image": 1}, seed=0)
        self.max_new_tokens = max_new_tokens

    def generate(self, rows, n=1, temperature=0.0, seed=0):
        from vllm import SamplingParams

        params = SamplingParams(n=n, temperature=temperature, max_tokens=self.max_new_tokens, seed=seed)
        inputs = []
        for r in rows:
            inputs.append({"prompt": chat_prompt(self.processor, user_messages(r)),
                           "multi_modal_data": {"image": load_image(r)}})
        outs = self.llm.generate(inputs, params)
        return [[o.text for o in out.outputs] for out in outs]


class HFBackend:
    def __init__(self, model, max_new_tokens, batch_size=8, adapter=None, **_):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def generate(self, rows, n=1, temperature=0.0, seed=0):
        self.torch.manual_seed(seed)
        results = []
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i:i + self.batch_size]
            texts = [chat_prompt(self.processor, user_messages(r)) for r in batch]
            images = [load_image(r) for r in batch]
            enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.model.device)
            gen = dict(max_new_tokens=self.max_new_tokens, num_return_sequences=n)
            gen.update(dict(do_sample=True, temperature=temperature) if temperature > 0 else dict(do_sample=False))
            with self.torch.no_grad():
                out = self.model.generate(**enc, **gen)
            out = out[:, enc["input_ids"].shape[1]:]
            decoded = self.processor.batch_decode(out, skip_special_tokens=True)
            results += [decoded[j * n:(j + 1) * n] for j in range(len(batch))]
            print(f"  {min(i + self.batch_size, len(rows))}/{len(rows)}", end="\r")
        print()
        return results


class APIBackend:
    """Needs OPENAI_API_KEY and OPENAI_BASE_URL (e.g. the DashScope
    OpenAI-compatible endpoint). Sequential; fine for ~1k test questions."""

    def __init__(self, model, max_new_tokens, **_):
        from openai import OpenAI

        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))
        self.model = model
        self.max_new_tokens = max_new_tokens

    def generate(self, rows, n=1, temperature=0.0, seed=0):
        results = []
        for k, r in enumerate(rows):
            with open(image_path(r), "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            msgs = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": user_text(r)}]}]
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, n=n, temperature=temperature,
                max_tokens=self.max_new_tokens, seed=seed)
            results.append([c.message.content or "" for c in resp.choices])
            print(f"  {k + 1}/{len(rows)}", end="\r")
        print()
        return results
