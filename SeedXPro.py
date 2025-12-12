from openai import OpenAI
import time
import json
import os
import re


class RN_SeedXPro_Translator():
    language_code_map = {
        "Arabic": "ar",
        "French": "fr",
        "Malay": "ms",
        "Russian": "ru",
        "Czech": "cs",
        "Croatian": "hr",
        "Norwegian Bokmal": "nb",
        "Swedish": "sv",
        "Danish": "da",
        "Hungarian": "hu",
        "Dutch": "nl",
        "Thai": "th",
        "German": "de",
        "Indonesian": "id",
        "Norwegian": "no",
        "Turkish": "tr",
        "English": "en",
        "Italian": "it",
        "Polish": "pl",
        "Ukrainian": "uk",
        "Spanish": "es",
        "Japanese": "ja",
        "Portuguese": "pt",
        "Vietnamese": "vi",
        "Finnish": "fi",
        "Korean": "ko",
        "Romanian": "ro",
        "Chinese": "zh"
    }

    def __init__(self):
        pass

    def _load_llm_config(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "comfyui_rn_translator-config.json")
        if not os.path.exists(cfg_path):
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            llm = data.get("llm") or {}
            current = llm.get("current_provider")
            providers = llm.get("providers") or {}
            provider_cfg = providers.get(current) or {}
            return provider_cfg
        except Exception:
            return {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "may the force be with you"}),
                "from": (list(cls.language_code_map.keys()), {'default': 'English'}),
                "to": (list(cls.language_code_map.keys()), {'default': 'Chinese'}),
            },
            "optional": {
                "seed": ("INT", {"default": 28, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("content",)
    FUNCTION = "translate"
    CATEGORY = "RunNode/RH_SeedXPro"
    TITLE = "RunNode SeedXPro Translator"

    def _split_text_into_chunks(self, text, max_chunk_size=400):
        if len(text) <= max_chunk_size:
            return [text]
        sentences = re.split(r'[.!?。！？]\s*', text)
        chunks = []
        current = ""
        for s in sentences:
            if len(current + s) <= max_chunk_size:
                current += (s + ". " if s else "")
            else:
                if current:
                    chunks.append(current.strip())
                current = (s + ". " if s else "")
        if current:
            chunks.append(current.strip())
        final = []
        for c in chunks:
            if len(c) <= max_chunk_size:
                final.append(c)
            else:
                for i in range(0, len(c), max_chunk_size):
                    final.append(c[i:i + max_chunk_size])
        return final

    def _translate_chunk(self, chunk, src, dst, temperature, apiBaseUrl, apiKey, model):
        if apiBaseUrl == "default":
            apiBaseUrl = ""
        if apiKey == "default":
            apiKey = ""
        if model == "default":
            model = ""
        env_api_baseurl = (
            os.environ.get("LLM_API_BASEURL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE_URL")
            or os.environ.get("DEEPSEEK_API_BASE_URL")
        )
        env_api_key = (
            os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
        )
        env_model = (
            os.environ.get("LLM_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("DEEPSEEK_MODEL")
        )

        cfg = self._load_llm_config()
        cfg_base_url = cfg.get("base_url")
        cfg_api_key = cfg.get("api_key")
        cfg_model = cfg.get("model")
        cfg_temperature = cfg.get("temperature")
        cfg_max_tokens = cfg.get("max_tokens")
        cfg_top_p = cfg.get("top_p")

        used_api_baseurl = (apiBaseUrl or env_api_baseurl or cfg_base_url or "https://api.openai.com/v1")
        used_model = (model or env_model or cfg_model or "gpt-4o-mini")
        used_api_key = (apiKey or env_api_key or cfg_api_key or "")
        if not used_api_key:
            return "错误：请提供API密钥"

        try:
            client = OpenAI(api_key=used_api_key, base_url=used_api_baseurl)
            system_prompt = "你是一个专业的翻译助手。"
            user_prompt = f"Translate from {src} to {dst}:\n{chunk}\n\nOnly return the translation in {dst}."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()},
            ]
            use_temperature = (
                temperature if temperature is not None else (
                    cfg_temperature if cfg_temperature is not None else 0.3
                )
            )
            use_max_tokens = cfg_max_tokens if cfg_max_tokens is not None else 1024
            params = {
                "model": used_model,
                "messages": messages,
                "temperature": use_temperature,
                "max_tokens": use_max_tokens,
            }
            if cfg_top_p is not None:
                params["top_p"] = cfg_top_p
            completion = client.chat.completions.create(**params)
            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                return completion.choices[0].message.content.strip()
            return "错误：API返回空结果"
        except Exception as e:
            return f"翻译错误：{str(e)}"

    def translate(self, prompt, **kwargs):
        cleaned = re.sub(r'[\x00\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', prompt or "")
        if not cleaned.strip():
            return ("错误：请输入要翻译的文本",)
        src = kwargs.get('from')
        dst = kwargs.get('to')

        chunks = self._split_text_into_chunks(cleaned, max_chunk_size=400)
        if len(chunks) == 1:
            res = self._translate_chunk(chunks[0], src, dst, None, None, None, None)
            return (res,)
        translated = []
        for c in chunks:
            translated.append(self._translate_chunk(c, src, dst, None, None, None, None))
        return (' '.join(translated),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())
