# CounterHarm-SHIELD: Dialect-Aware Benign Sample Generation
# Sub-module of Dia-LLM within the DIA-GUARD framework
#
# SHIELD = Safety Harm Identification in English Language Dialects
#
# Pipeline (6-chain CoI):
#   Chain 1 — ToxiCraft attribute extraction
#   Chain 2 — ToxiCraft benign prompt construction
#   Chain 3 — ToxiCraft contextual anchoring (CAE)
#   Chain 4 — ToxiCraft thematic style refinement (TSR)
#   Chain 5 — PromptSafe gated harmlessness scoring
#   Chain 6 — FIZLE counterfactual label validation
#
# Usage:
#   from CounterHarm_Shield.counterharm_pipeline import CounterHarmSHIELD
#   pipeline = CounterHarmSHIELD(llm=llm)
#   state = pipeline.run("harmful text here", dialect="urban_aave")

from .counterharm_pipeline import CounterHarmSHIELD, CounterHarmChainState

__version__ = "1.0.0"
__all__ = ["CounterHarmSHIELD", "CounterHarmChainState"]
