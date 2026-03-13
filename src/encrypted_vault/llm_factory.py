"""LLM provider factory — swap OpenAI / Anthropic / Ollama via config."""

from enum import Enum

from langchain_core.language_models import BaseChatModel


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMFactory:
    """
    Static factory for creating LangChain chat model instances.

    All agents use the same interface (BaseChatModel) regardless of provider.
    To switch providers, change LLM_PROVIDER in .env — no code changes needed.
    """

    @staticmethod
    def create(
        provider: LLMProvider = LLMProvider.OPENAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        **kwargs,
    ) -> BaseChatModel:
        """
        Instantiate a chat model for the given provider.

        Args:
            provider: Which LLM backend to use.
            model: Model name/identifier (provider-specific).
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            **kwargs: Additional provider-specific kwargs.

        Returns:
            A LangChain BaseChatModel instance ready for tool binding.
        """
        match provider:
            case LLMProvider.OPENAI:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(model=model, temperature=temperature, **kwargs)

            case LLMProvider.ANTHROPIC:
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(model=model, temperature=temperature, **kwargs)

            case LLMProvider.OLLAMA:
                from langchain_ollama import ChatOllama

                return ChatOllama(model=model, temperature=temperature, **kwargs)

            case _:
                raise ValueError(f"Unsupported LLM provider: {provider!r}")

    @staticmethod
    def create_default() -> BaseChatModel:
        """
        Create the default LLM using settings from config.
        All 4 agents use this by default (gpt-4o-mini).
        Explicitly passes the API key so it works regardless of whether
        the OS environment variable is set.
        """
        from encrypted_vault.config import settings

        return LLMFactory.create(
            provider=LLMProvider.OPENAI,
            model=settings.llm_model,
            temperature=0.7,
            api_key=settings.openai_api_key,
        )
