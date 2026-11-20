"""The pipeline core: Steps, Pipelines, Resources, and the run engine.

The pipeline owns only the DAG; artifacts, their paths, serializers and
completeness live in :mod:`gnomad_meb.lib.checkpoint`, which this package does
not import at load time. :class:`~gnomad_meb.pipeline.PipelineFactory` imports
the stage modules lazily, so importing the core does not drag in the
hail-bearing library.
"""

from __future__ import annotations

from .constants import RUNTIME_PLACEHOLDER
from .execution import PipelineMismatchError, PipelineRunner, run_pipeline
from .factory import PipelineFactory
from .markers import FromStep, ResourceRef, RunInput
from .pipeline import Pipeline
from .registry import ReferenceGenomeMismatch, ResourceRegistry
from .report import (
    RecordingObserver,
    RunObserver,
    TerminalObserver,
    format_duration,
    render_plan,
    render_step_line,
    render_summary,
)
from .resource import Resource
from .runtime import LocalRuntime, StepContext, StepRuntime
from .spec import InputStep, RunManifest, Step, StepState
from .status import StepResult, StepStatus
from .step import StepRegistry

__all__ = [
    "RUNTIME_PLACEHOLDER",
    "FromStep",
    "InputStep",
    "LocalRuntime",
    "Pipeline",
    "PipelineFactory",
    "PipelineMismatchError",
    "PipelineRunner",
    "RecordingObserver",
    "ReferenceGenomeMismatch",
    "Resource",
    "ResourceRef",
    "ResourceRegistry",
    "RunInput",
    "RunManifest",
    "RunObserver",
    "Step",
    "StepContext",
    "StepRegistry",
    "StepResult",
    "StepRuntime",
    "StepState",
    "StepStatus",
    "TerminalObserver",
    "format_duration",
    "render_plan",
    "render_step_line",
    "render_summary",
    "run_pipeline",
]
