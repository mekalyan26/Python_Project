"""
LLM Configuration Utility
Unified interface for OpenAI, AWS Bedrock and Hugging Face.
High-level selector via AI_STACK env (openai | aws | huggingface).
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    AZURE = "azure"       # kept for compatibility
    ANTHROPIC = "anthropic"  # kept for compatibility


@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    # Azure
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    # Bedrock
    bedrock_region: Optional[str] = None
    # Hugging Face
    hf_token: Optional[str] = None
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class EmbeddingConfig:
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    dimension: int = 1536
    # Bedrock
    bedrock_region: Optional[str] = None
    # Azure
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    # Hugging Face
    hf_token: Optional[str] = None


class LLMConfigUtil:

    @staticmethod
    def _resolve_stack_defaults() -> Dict[str, Any]:
        """Map AI_STACK -> default providers and models."""
        stack = os.getenv("AI_STACK", "openai").strip().lower()
        if stack not in ("openai", "aws", "huggingface"):
            stack = "openai"

        if stack == "aws":
            return {
                "llm_provider": LLMProvider.BEDROCK,
                "llm_model": os.getenv("LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"),
                "embed_provider": LLMProvider.BEDROCK,
                "embed_model": os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v1"),
                "embed_dim": int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            }
        if stack == "huggingface":
            return {
                "llm_provider": LLMProvider.HUGGINGFACE,
                "llm_model": os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
                "embed_provider": LLMProvider.HUGGINGFACE,
                "embed_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                "embed_dim": int(os.getenv("EMBEDDING_DIMENSION", "384")),
            }
        # openai (default)
        return {
            "llm_provider": LLMProvider.OPENAI,
            "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "embed_provider": LLMProvider.OPENAI,
            "embed_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            "embed_dim": int(os.getenv("EMBEDDING_DIMENSION", "1536")),
        }

    @staticmethod
    def load_llm_config() -> LLMConfig:
        stack_defaults = LLMConfigUtil._resolve_stack_defaults()

        provider = stack_defaults["llm_provider"]
        model = stack_defaults["llm_model"]

        cfg = LLMConfig(
            provider=provider,
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
            top_p=float(os.getenv("LLM_TOP_P", "1.0")),
        )

        if provider == LLMProvider.BEDROCK:
            cfg.bedrock_region = os.getenv("BEDROCK_REGION", "us-east-1")
        elif provider == LLMProvider.HUGGINGFACE:
            cfg.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        elif provider == LLMProvider.AZURE:
            cfg.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            cfg.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            cfg.api_key = os.getenv("AZURE_OPENAI_API_KEY")

        logger.info(f"✓ LLM config: stack={os.getenv('AI_STACK','openai')} provider={cfg.provider.value} model={cfg.model}")
        return cfg

    @staticmethod
    def load_embedding_config() -> EmbeddingConfig:
        stack_defaults = LLMConfigUtil._resolve_stack_defaults()
        provider = stack_defaults["embed_provider"]
        model = stack_defaults["embed_model"]

        cfg = EmbeddingConfig(
            provider=provider,
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            dimension=stack_defaults["embed_dim"],
        )

        if provider == LLMProvider.BEDROCK:
            cfg.bedrock_region = os.getenv("BEDROCK_REGION", "us-east-1")
        elif provider == LLMProvider.HUGGINGFACE:
            cfg.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        elif provider == LLMProvider.AZURE:
            cfg.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            cfg.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            cfg.api_key = os.getenv("AZURE_OPENAI_API_KEY")

        logger.info(f"✓ Embedding config: provider={cfg.provider.value} model={cfg.model} dim={cfg.dimension}")
        return cfg

    @staticmethod
    def get_llm_client(config: LLMConfig):
        if config.provider == LLMProvider.OPENAI:
            from openai import OpenAI
            return OpenAI(api_key=config.api_key)
        if config.provider == LLMProvider.BEDROCK:
            import boto3
            return boto3.client("bedrock-runtime", region_name=config.bedrock_region)
        if config.provider == LLMProvider.HUGGINGFACE:
            from huggingface_hub import InferenceClient
            return InferenceClient(model=config.model, token=config.hf_token)
        if config.provider == LLMProvider.AZURE:
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.azure_endpoint,
                api_version="2024-02-15-preview"
            )
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    @staticmethod
    def get_embedding_client(config: EmbeddingConfig):
        if config.provider == LLMProvider.OPENAI:
            from openai import OpenAI
            return OpenAI(api_key=config.api_key)
        if config.provider == LLMProvider.BEDROCK:
            import boto3
            return boto3.client("bedrock-runtime", region_name=config.bedrock_region)
        if config.provider == LLMProvider.HUGGINGFACE:
            # Use SentenceTransformer for robust vector size
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(config.model)
        if config.provider == LLMProvider.AZURE:
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.azure_endpoint,
                api_version="2024-02-15-preview"
            )
        raise ValueError(f"Unsupported embedding provider: {config.provider}")

    @staticmethod
    def generate_completion(config: LLMConfig, messages: List[Dict[str, str]], **kwargs) -> str:
        client = LLMConfigUtil.get_llm_client(config)
        params = {
            "model": kwargs.get("model", config.model),
            "temperature": kwargs.get("temperature", config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "top_p": kwargs.get("top_p", config.top_p),
        }

        # AWS Bedrock
        if config.provider == LLMProvider.BEDROCK:
            try:
                system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
                user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
                if "anthropic.claude" in params["model"]:
                    body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": params["max_tokens"],
                        "temperature": params["temperature"],
                        "top_p": params["top_p"],
                        "system": system_text,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
                    }
                else:
                    prompt = (system_text + "\n\n" + user_text).strip()
                    body = {
                        "inputText": prompt,
                        "textGenerationConfig": {
                            "temperature": params["temperature"],
                            "topP": params["top_p"],
                            "maxTokenCount": params["max_tokens"],
                        },
                    }
                resp = client.invoke_model(
                    modelId=params["model"],
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                raw = json.loads(resp["body"].read())
                if "content" in raw:  # Anthropic
                    return "".join(part.get("text", "") for part in raw["content"])
                if "outputText" in raw:  # Titan style
                    return raw["outputText"]
                if "results" in raw and raw["results"]:
                    return raw["results"][0].get("outputText", "")
                return json.dumps(raw)
            except Exception as e:
                logger.exception("Bedrock generation failed")
                raise

        # Hugging Face (Inference API)
        if config.provider == LLMProvider.HUGGINGFACE:
            # Concatenate messages to a single prompt
            system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
            user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
            prompt = (system_text + "\n\n" + user_text).strip()
            try:
                # Prefer chat.completions if available; else text_generation
                if hasattr(client, "chat_completions"):
                    out = client.chat_completions.create(
                        messages=messages,
                        model=params["model"],
                        max_tokens=params["max_tokens"],
                        temperature=params["temperature"],
                        top_p=params["top_p"],
                    )
                    return out.choices[0].message.content
                else:
                    out = client.text_generation(
                        prompt=prompt,
                        max_new_tokens=params["max_tokens"],
                        temperature=params["temperature"],
                        top_p=params["top_p"],
                        stream=False,
                    )
                    # text_generation returns str
                    return out
            except Exception as e:
                logger.exception("Hugging Face generation failed")
                raise

        # OpenAI / Azure OpenAI
        if config.provider == LLMProvider.AZURE and config.azure_deployment:
            params["model"] = config.azure_deployment
        response = client.chat.completions.create(messages=messages, **params)
        return response.choices[0].message.content

    @staticmethod
    def generate_embeddings(config: EmbeddingConfig, texts: List[str]) -> List[List[float]]:
        client = LLMConfigUtil.get_embedding_client(config)

        # Bedrock Titan embeddings
        if config.provider == LLMProvider.BEDROCK:
            vectors: List[List[float]] = []
            for t in texts:
                body = {"inputText": t}
                resp = client.invoke_model(
                    modelId=config.model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                data = json.loads(resp["body"].read())
                emb = data.get("embedding") or data.get("vector") or []
                if not emb:
                    logger.warning("Bedrock returned empty embedding.")
                vectors.append(emb)
            return vectors

        # Hugging Face (SentenceTransformers)
        if config.provider == LLMProvider.HUGGINGFACE:
            # client is SentenceTransformer model instance
            vectors = client.encode(texts, normalize_embeddings=True)
            # Ensure Python list of lists
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

        # OpenAI / Azure
        model = config.azure_deployment if config.provider == LLMProvider.AZURE else config.model
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]


# Convenience
def get_default_llm_config() -> LLMConfig:
    return LLMConfigUtil.load_llm_config()


def get_default_embedding_config() -> EmbeddingConfig:
    return LLMConfigUtil.load_embedding_config()


def generate_text(messages: List[Dict[str, str]], **kwargs) -> str:
    cfg = get_default_llm_config()
    return LLMConfigUtil.generate_completion(cfg, messages, **kwargs)