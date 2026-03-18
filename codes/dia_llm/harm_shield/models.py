# models.py - LLM backend implementations for OpenAI API and local/open-source models

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """Generate a response from the model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend."""
        pass


# ═══════════════════════════════════════════════════════════════════
# OPTION 1: OpenAI API
# ═══════════════════════════════════════════════════════════════════

class OpenAIBackend(BaseLLM):
    """OpenAI API backend (GPT-4, GPT-4 Turbo, GPT-3.5 Turbo)."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # For Azure or compatible APIs
        organization: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.organization = organization
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                if self.organization:
                    kwargs["organization"] = self.organization
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        try:
            from openai import OpenAI
            return self.api_key is not None
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"


# ═══════════════════════════════════════════════════════════════════
# OPTION 1B: Azure OpenAI API
# ═══════════════════════════════════════════════════════════════════

class AzureOpenAIBackend(BaseLLM):
    """Azure OpenAI API backend.

    Supports Azure OpenAI Service deployments using either:
    1. The official Azure OpenAI SDK (AzureOpenAI client)
    2. OpenAI SDK with custom base_url for /openai/v1 endpoints

    Usage:
        # Option 1: Using Azure-specific parameters
        llm = AzureOpenAIBackend(
            deployment_name="gpt-4.1",
            endpoint="https://your-resource.openai.azure.com",
            api_key="your-api-key",
            api_version="2024-02-15-preview"
        )

        # Option 2: Using base_url (for /openai/v1 compatible endpoints)
        llm = AzureOpenAIBackend(
            deployment_name="gpt-4.1",
            base_url="https://your-resource.openai.azure.com/openai/v1",
            api_key="your-api-key"
        )
    """

    def __init__(
        self,
        deployment_name: str,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        use_azure_client: bool = True
    ):
        """
        Initialize Azure OpenAI backend.

        Args:
            deployment_name: The name of your Azure OpenAI deployment
            endpoint: Azure OpenAI endpoint (e.g., "https://your-resource.openai.azure.com")
            base_url: Alternative: direct base_url for OpenAI-compatible endpoints
            api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)
            api_version: Azure API version (default: 2024-02-15-preview)
            use_azure_client: If True, use AzureOpenAI client; if False, use OpenAI with base_url
        """
        self.deployment_name = deployment_name
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.base_url = base_url
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.use_azure_client = use_azure_client
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI, AzureOpenAI
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai>=1.0.0")

            # If base_url is provided, use OpenAI client with custom base_url
            if self.base_url:
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
            elif self.use_azure_client and self.endpoint:
                # Use official AzureOpenAI client
                self._client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version
                )
            else:
                raise ValueError(
                    "Must provide either 'base_url' or 'endpoint' for Azure OpenAI. "
                    "Set endpoint via AZURE_OPENAI_ENDPOINT env var or pass directly."
                )
        return self._client

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        # Newer models (gpt-5.x, o1, o3) use max_completion_tokens and don't support temperature
        is_new_model = any(m in self.deployment_name for m in ["gpt-5", "o1", "o3"])
        extra_params = {}
        if is_new_model:
            extra_params["max_completion_tokens"] = max_tokens
        else:
            extra_params["max_tokens"] = max_tokens
            extra_params["temperature"] = temperature

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            **extra_params
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        try:
            from openai import OpenAI
            return self.api_key is not None and (self.endpoint is not None or self.base_url is not None)
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return f"Azure OpenAI ({self.deployment_name})"


# ═══════════════════════════════════════════════════════════════════
# OPTION 1C: Google Gemini via AI Studio (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════

class GeminiBackend(BaseLLM):
    """Google Gemini backend via Vertex AI or Google AI Studio.

    Uses the google-genai SDK (pip install google-genai) with Vertex AI
    for production workloads (higher rate limits, safety settings control).
    Falls back to AI Studio (API key) or OpenAI-compatible endpoint.

    Usage:
        # Vertex AI (recommended - requires: gcloud auth application-default login)
        llm = GeminiBackend(
            model="gemini-3.1-flash-lite-preview",
            use_vertex=True,
            project_id="diaguard-new-project",
            location="us-central1"
        )

        # AI Studio (API key)
        llm = GeminiBackend(
            model="gemini-3.1-flash-lite-preview",
            api_key="your-api-key"
        )
    """

    # Thread-safe rate limiter
    _rate_lock = None
    _last_call_time = 0
    _MIN_INTERVAL = 0.5  # 0.5 seconds between calls (Vertex AI has higher limits)

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite-preview",
        api_key: Optional[str] = None,
        use_vertex: bool = True,
        project_id: str = "diaguard-new-project",
        location: str = "us-central1"
    ):
        import threading
        if GeminiBackend._rate_lock is None:
            GeminiBackend._rate_lock = threading.Lock()
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.use_vertex = use_vertex
        self.project_id = project_id
        self.location = location
        self._client = None
        self._use_native = None
        self._safety_settings = None

    def _build_safety_settings(self):
        """Build safety settings with BLOCK_NONE for all categories."""
        from google.genai import types
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
        ]

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
                if self.use_vertex:
                    # Vertex AI: uses gcloud auth application-default login
                    self._client = genai.Client(
                        vertexai=True,
                        project=self.project_id,
                        location=self.location
                    )
                    self._use_native = True
                    self._safety_settings = self._build_safety_settings()
                    print(f"[Gemini] Connected via Vertex AI (Project: {self.project_id}, Location: {self.location})")
                else:
                    # AI Studio: uses API key
                    self._client = genai.Client(api_key=self.api_key)
                    self._use_native = True
                    self._safety_settings = self._build_safety_settings()
            except ImportError:
                # Fall back to OpenAI-compatible endpoint
                try:
                    from openai import OpenAI
                    self._client = OpenAI(
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                        api_key=self.api_key
                    )
                    self._use_native = False
                except ImportError:
                    raise ImportError(
                        "Neither google-genai nor openai package installed. "
                        "Run: pip install google-genai  OR  pip install openai>=1.0.0"
                    )
        return self._client

    def _wait_for_rate_limit(self):
        """Enforce rate limit by waiting if needed."""
        import time
        with GeminiBackend._rate_lock:
            now = time.time()
            elapsed = now - GeminiBackend._last_call_time
            if elapsed < self._MIN_INTERVAL:
                wait = self._MIN_INTERVAL - elapsed
                time.sleep(wait)
            GeminiBackend._last_call_time = time.time()

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        _ = self.client  # Ensure client is initialized
        self._wait_for_rate_limit()

        if self._use_native:
            # Native google-genai SDK (Vertex AI or AI Studio)
            from google.genai import types
            config_kwargs = {
                "system_instruction": system,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if self._safety_settings:
                config_kwargs["safety_settings"] = self._safety_settings

            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(**config_kwargs)
            )
            return response.text
        else:
            # OpenAI-compatible fallback
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

    def is_available(self) -> bool:
        if self.use_vertex:
            try:
                from google import genai
                return True  # Vertex AI uses gcloud auth, no API key needed
            except ImportError:
                return False
        try:
            from google import genai
            return self.api_key is not None
        except ImportError:
            try:
                from openai import OpenAI
                return self.api_key is not None
            except ImportError:
                return False

    @property
    def name(self) -> str:
        mode = "Vertex AI" if self.use_vertex else "AI Studio"
        return f"Gemini ({self.model}) [{mode}]"


# ═══════════════════════════════════════════════════════════════════
# OPTION 2A: Ollama (Local Models)
# ═══════════════════════════════════════════════════════════════════

class OllamaBackend(BaseLLM):
    """Ollama backend for local models (Llama, Mistral, etc.)."""

    def __init__(
        self,
        model: str = "llama3.1",  # or mistral, mixtral, phi3, etc.
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise ImportError("Ollama package not installed. Run: pip install ollama")
        return self._client

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response["message"]["content"]

    def is_available(self) -> bool:
        try:
            import ollama
            # Try to list models to check if Ollama is running
            client = ollama.Client(host=self.host)
            client.list()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"Ollama ({self.model})"


# ═══════════════════════════════════════════════════════════════════
# OPTION 2B: Hugging Face Transformers (Local Models)
# ═══════════════════════════════════════════════════════════════════

class HuggingFaceBackend(BaseLLM):
    """Hugging Face Transformers backend for local models."""

    # Recommended models for dialect transformation
    RECOMMENDED_MODELS = {
        "small": "microsoft/Phi-3-mini-4k-instruct",  # ~4GB VRAM
        "medium": "mistralai/Mistral-7B-Instruct-v0.3",  # ~16GB VRAM
        "large": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # ~18GB VRAM
        "xlarge": "meta-llama/Meta-Llama-3.1-70B-Instruct"  # ~140GB VRAM (quantized: ~40GB)
    }

    def __init__(
        self,
        model: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "auto",  # "auto", "cuda", "cpu", "mps"
        torch_dtype: str = "auto",  # "auto", "float16", "bfloat16", "float32"
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True
    ):
        self.model_name = model
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None
        self._pipeline = None

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "Transformers package not installed. Run: "
                "pip install transformers torch accelerate bitsandbytes"
            )

        # Determine torch dtype
        if self.torch_dtype == "auto":
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            dtype = dtype_map.get(self.torch_dtype, torch.float32)

        # Quantization config
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )

        # Load model
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = dtype

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # Create pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer
        )

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        self._load_model()

        # Format as chat messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        # Generate
        outputs = self._pipeline(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.01,
            do_sample=temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id
        )

        # Extract generated text
        generated = outputs[0]["generated_text"]

        # Return only the assistant's response
        if isinstance(generated, list):
            # Chat format returns list of messages
            for msg in reversed(generated):
                if msg.get("role") == "assistant":
                    return msg["content"]
            return generated[-1]["content"] if generated else ""
        else:
            # Legacy format returns string
            return generated

    def is_available(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return f"HuggingFace ({self.model_name.split('/')[-1]})"


# ═══════════════════════════════════════════════════════════════════
# OPTION 2C: vLLM (High-Performance Local Inference)
# ═══════════════════════════════════════════════════════════════════

class VLLMBackend(BaseLLM):
    """vLLM backend for high-performance local inference."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096
    ):
        self.model_name = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None
        self._tokenizer = None

    def _load_model(self):
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "vLLM package not installed. Run: pip install vllm"
            )

        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        self._load_model()
        from vllm import SamplingParams

        # Format as chat
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.01,
            max_tokens=max_tokens
        )

        outputs = self._llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def is_available(self) -> bool:
        try:
            from vllm import LLM
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return f"vLLM ({self.model_name.split('/')[-1]})"


# ═══════════════════════════════════════════════════════════════════
# OPTION 3: Anthropic Claude API
# ═══════════════════════════════════════════════════════════════════

class AnthropicBackend(BaseLLM):
    """Anthropic Claude API backend."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        return self._client

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[
                {"role": "user", "content": user}
            ],
            temperature=temperature
        )
        return response.content[0].text

    def is_available(self) -> bool:
        try:
            import anthropic
            return self.api_key is not None
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"


# ═══════════════════════════════════════════════════════════════════
# Model Factory and Auto-Detection
# ═══════════════════════════════════════════════════════════════════

class ModelFactory:
    """Factory for creating LLM backends with auto-detection."""

    BACKENDS = {
        "openai": OpenAIBackend,
        "azure": AzureOpenAIBackend,
        "azure_openai": AzureOpenAIBackend,
        "gemini": GeminiBackend,
        "ollama": OllamaBackend,
        "huggingface": HuggingFaceBackend,
        "vllm": VLLMBackend,
        "anthropic": AnthropicBackend
    }

    @classmethod
    def create(
        cls,
        backend: str = "auto",
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM backend.

        Args:
            backend: One of "openai", "azure", "ollama", "huggingface", "vllm", "anthropic", or "auto"
            **kwargs: Backend-specific configuration

        Returns:
            Configured LLM backend
        """
        if backend == "auto":
            return cls.auto_detect(**kwargs)

        if backend not in cls.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(cls.BACKENDS.keys())}")

        return cls.BACKENDS[backend](**kwargs)

    @classmethod
    def auto_detect(cls, **kwargs) -> BaseLLM:
        """Auto-detect and return the best available backend."""

        # Priority order: OpenAI > Azure OpenAI > Anthropic > Ollama > HuggingFace

        # Check OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                backend = OpenAIBackend(**kwargs)
                if backend.is_available():
                    print(f"[ModelFactory] Auto-detected: {backend.name}")
                    return backend
            except Exception:
                pass

        # Check Azure OpenAI
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            try:
                backend = AzureOpenAIBackend(
                    deployment_name=kwargs.get("deployment_name", kwargs.get("model", "gpt-4")),
                    **{k: v for k, v in kwargs.items() if k not in ["deployment_name", "model"]}
                )
                if backend.is_available():
                    print(f"[ModelFactory] Auto-detected: {backend.name}")
                    return backend
            except Exception:
                pass

        # Check Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                backend = AnthropicBackend(**kwargs)
                if backend.is_available():
                    print(f"[ModelFactory] Auto-detected: {backend.name}")
                    return backend
            except Exception:
                pass

        # Check Ollama
        try:
            backend = OllamaBackend(**kwargs)
            if backend.is_available():
                print(f"[ModelFactory] Auto-detected: {backend.name}")
                return backend
        except Exception:
            pass

        # Fall back to HuggingFace
        try:
            backend = HuggingFaceBackend(**kwargs)
            if backend.is_available():
                print(f"[ModelFactory] Auto-detected: {backend.name}")
                return backend
        except Exception:
            pass

        raise RuntimeError(
            "No LLM backend available. Please either:\n"
            "1. Set OPENAI_API_KEY environment variable\n"
            "2. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables\n"
            "3. Set ANTHROPIC_API_KEY environment variable\n"
            "4. Install and run Ollama (https://ollama.ai)\n"
            "5. Install transformers: pip install transformers torch"
        )

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available backends."""
        available = []

        for name, backend_cls in cls.BACKENDS.items():
            try:
                backend = backend_cls()
                if backend.is_available():
                    available.append(f"{name}: {backend.name}")
            except Exception:
                pass

        return available


# ═══════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════

def get_openai_backend(
    model: str = "gpt-4",
    api_key: Optional[str] = None
) -> OpenAIBackend:
    """Create an OpenAI backend."""
    return OpenAIBackend(model=model, api_key=api_key)


def get_ollama_backend(
    model: str = "llama3.1",
    host: str = "http://localhost:11434"
) -> OllamaBackend:
    """Create an Ollama backend."""
    return OllamaBackend(model=model, host=host)


def get_huggingface_backend(
    model: str = "microsoft/Phi-3-mini-4k-instruct",
    load_in_4bit: bool = True
) -> HuggingFaceBackend:
    """Create a HuggingFace backend with 4-bit quantization."""
    return HuggingFaceBackend(model=model, load_in_4bit=load_in_4bit)


def get_anthropic_backend(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None
) -> AnthropicBackend:
    """Create an Anthropic backend."""
    return AnthropicBackend(model=model, api_key=api_key)


def get_gemini_backend(
    model: str = "gemini-3.1-flash-lite-preview",
    api_key: Optional[str] = None,
    use_vertex: bool = True,
    project_id: str = "diaguard-new-project",
    location: str = "us-central1"
) -> GeminiBackend:
    """Create a Google Gemini backend via Vertex AI or AI Studio."""
    return GeminiBackend(
        model=model,
        api_key=api_key,
        use_vertex=use_vertex,
        project_id=project_id,
        location=location,
    )


def get_azure_openai_backend(
    deployment_name: str,
    endpoint: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: str = "2024-02-15-preview"
) -> AzureOpenAIBackend:
    """
    Create an Azure OpenAI backend.

    Args:
        deployment_name: Your Azure OpenAI deployment name (e.g., "gpt-4.1")
        endpoint: Azure endpoint (e.g., "https://your-resource.openai.azure.com")
        base_url: Alternative: direct base_url for /openai/v1 compatible endpoints
        api_key: Azure OpenAI API key
        api_version: Azure API version

    Example:
        # Using endpoint (recommended)
        llm = get_azure_openai_backend(
            deployment_name="gpt-4.1",
            endpoint="https://jsl-diaguard.openai.azure.com",
            api_key="your-api-key"
        )

        # Using base_url
        llm = get_azure_openai_backend(
            deployment_name="gpt-4.1",
            base_url="https://jsl-diaguard.openai.azure.com/openai/v1",
            api_key="your-api-key"
        )
    """
    return AzureOpenAIBackend(
        deployment_name=deployment_name,
        endpoint=endpoint,
        base_url=base_url,
        api_key=api_key,
        api_version=api_version
    )
