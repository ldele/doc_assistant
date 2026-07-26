"""Settings wire models (ADR-011 U1c, ADR-027 D2/E3).

Write side only. The *read* view is assembled in ``apps.api.services._settings_view`` from the
locked ``config`` values plus the live provider list, so it has no pydantic model here — the
locked knobs are read-only by design (they move via an eval-harness experiment, never a PATCH).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SettingsUpdate(BaseModel):
    """User-settable settings: the source documents folder, (ADR-011, U1c) the LLM
    provider/model to switch to, and (ADR-027 D2, E3) the persisted answer-layer epistemics
    toggle. At least one field must be present; the two ``llm_*`` fields travel together or not
    at all — both requests reject an otherwise-empty or half-shaped body with a 422, not a
    silent no-op. ``epistemics_markers_enabled`` governs the answer surface only (D2); the D3
    source-evaluation strip is always-on and has no setting by design."""

    source_dir: str | None = None
    llm_provider: Literal["anthropic", "ollama"] | None = None
    llm_model: str | None = Field(default=None, min_length=1)
    epistemics_markers_enabled: bool | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> SettingsUpdate:
        if (
            self.source_dir is None
            and self.llm_provider is None
            and self.llm_model is None
            and self.epistemics_markers_enabled is None
        ):
            raise ValueError(
                "at least one of source_dir, llm_provider+llm_model, or "
                "epistemics_markers_enabled is required"
            )
        if (self.llm_provider is None) != (self.llm_model is None):
            raise ValueError("llm_provider and llm_model must be provided together")
        return self
