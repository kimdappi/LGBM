from __future__ import annotations

from typing import Dict

from .runner import ToolRegistry
from .tools.behavior_topk_direct_compare import BehaviorTopKDirectCompareTool
from .tools.lens_diagnostic_consistency import LensDiagnosticConsistencyTool
from .tools.lens_monitoring_response import LensMonitoringResponseTool
from .tools.lens_severity_risk import LensSeverityRiskTool
from .tools.preprocess_evidence import PreprocessEvidenceTool
from .tools.preprocess_gaps import PreprocessRecordGapTool
from .tools.preprocess_timeline import PreprocessTimelineTool


def build_default_registry() -> ToolRegistry:
    tools = [
        # preprocessing (always)
        PreprocessTimelineTool(),
        PreprocessEvidenceTool(),
        PreprocessRecordGapTool(),
        # lens (specialized)
        LensSeverityRiskTool(),
        LensDiagnosticConsistencyTool(),
        LensMonitoringResponseTool(),
        # behavior (evidence strengthening)
        BehaviorTopKDirectCompareTool(),
    ]
    return ToolRegistry(tools={t.name: t for t in tools})

