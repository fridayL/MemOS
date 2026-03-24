from datetime import datetime
from typing import Any, ClassVar

from pydantic import ConfigDict, Field, field_validator, model_validator

from memos.configs.base import BaseConfig
from memos.configs.chunker import ChunkerConfigFactory
from memos.configs.embedder import EmbedderConfigFactory
from memos.configs.llm import LLMConfigFactory


class BaseMemReaderConfig(BaseConfig):
    """Base configuration class for MemReader."""

    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp for the MemReader"
    )

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_datetime(cls, value):
        """Parse datetime from string if needed."""
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    llm: LLMConfigFactory = Field(
        ..., description="LLM configuration for chat/doc memory extraction (fine-tuned model)"
    )
    general_llm: LLMConfigFactory | None = Field(
        default=None,
        description="General LLM for non-chat/doc tasks: hallucination filter, memory rewrite, "
        "memory merge, tool trajectory, skill memory. Falls back to main llm if not set.",
    )
    image_parser_llm: LLMConfigFactory | None = Field(
        default=None,
        description="Vision LLM for image parsing. Falls back to general_llm if not set.",
    )
    embedder: EmbedderConfigFactory = Field(
        ..., description="Embedder configuration for the MemReader"
    )
    chunker: ChunkerConfigFactory = Field(
        ..., description="Chunker configuration for the MemReader"
    )
    remove_prompt_example: bool = Field(
        default=False,
        description="whether remove example in memory extraction prompt to save token",
    )

    chat_chunker: dict[str, Any] = Field(
        default=None, description="Configuration for the MemReader chat chunk strategy"
    )


class SimpleStructMemReaderConfig(BaseMemReaderConfig):
    """SimpleStruct MemReader configuration class."""

    # Allow passing additional fields without raising validation errors
    model_config = ConfigDict(extra="allow", strict=True)


class MultiModalStructMemReaderConfig(BaseMemReaderConfig):
    """MultiModalStruct MemReader configuration class."""

    direct_markdown_hostnames: list[str] | None = Field(
        default=None,
        description="List of hostnames that should return markdown directly without parsing. "
        "If None, reads from FILE_PARSER_DIRECT_MARKDOWN_HOSTNAMES environment variable.",
    )

    oss_config: dict[str, Any] | None = Field(
        default=None,
        description="OSS configuration for the MemReader",
    )
    skills_dir_config: dict[str, Any] | None = Field(
        default=None,
        description="Skills directory for the MemReader",
    )


class StrategyStructMemReaderConfig(BaseMemReaderConfig):
    """StrategyStruct MemReader configuration class."""

    model_config = ConfigDict(extra="allow", strict=True)


class ToolAgentMemReaderConfig(MultiModalStructMemReaderConfig):
    """
    ToolAgent MemReader configuration.

    Extends MultiModalStructMemReaderConfig with additional 4B ReAct service parameters.
    Chat-type string_fine extraction is handled by a 4B model via HTTP tool calling API;
    all other logic (doc processing, skill/tool/pref extraction, embedding) is inherited
    from MultiModalStructMemReader.
    """

    api_url: str = Field(..., description="HTTP endpoint for the 4B tool calling service")
    api_key: str = Field(default="EMPTY", description="API key / Bearer token for the service")
    model: str = Field(default="qwen3-4B", description="Model name to send to the service")
    enable_thinking: bool = Field(default=True, description="Enable chain-of-thought / thinking mode")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens per API call")
    max_rounds: int = Field(default=6, description="Maximum tool-calling rounds per ReAct loop")
    search_top_k: int = Field(default=5, description="Top-K results returned by the search_memory tool")


class MemReaderConfigFactory(BaseConfig):
    """Factory class for creating MemReader configurations."""

    backend: str = Field(..., description="Backend for MemReader")
    config: dict[str, Any] = Field(..., description="Configuration for the MemReader backend")

    backend_to_class: ClassVar[dict[str, Any]] = {
        "simple_struct": SimpleStructMemReaderConfig,
        "multimodal_struct": MultiModalStructMemReaderConfig,
        "strategy_struct": StrategyStructMemReaderConfig,
        "tool_agent": ToolAgentMemReaderConfig,
    }

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, backend: str) -> str:
        """Validate the backend field."""
        if backend not in cls.backend_to_class:
            raise ValueError(f"Invalid backend: {backend}")
        return backend

    @model_validator(mode="after")
    def create_config(self) -> "MemReaderConfigFactory":
        config_class = self.backend_to_class[self.backend]
        self.config = config_class(**self.config)
        return self
