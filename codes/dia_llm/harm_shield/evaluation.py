# evaluation.py - LLM-as-a-Judge Evaluation for Dialect Transformations
#
# Provides multi-dimensional quality assessment of dialect transformations:
# - Fluency: Grammar, syntax, naturalness in target dialect
# - Faithfulness: Meaning preservation from original
# - Dialect Authenticity: Accuracy of dialect features
# - Feature Accuracy: Alignment with eWAVE linguistic feature specifications
# - Coherence: Logical flow and consistency
# - Readability: Ease of comprehension
#
# Supports: OpenAI, Anthropic, Ollama, HuggingFace backends

import re
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class EvaluationDimension(Enum):
    """Available evaluation dimensions."""
    FLUENCY = "fluency"
    FAITHFULNESS = "faithfulness"
    DIALECT_AUTHENTICITY = "dialect_authenticity"
    FEATURE_ACCURACY = "feature_accuracy"
    COHERENCE = "coherence"
    READABILITY = "readability"
    OVERALL = "overall"


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    dimension: str
    score: int  # 1-7 scale
    reasoning: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    # Additional fields for feature_accuracy dimension
    features_present: List[str] = field(default_factory=list)
    features_missing: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a dialect transformation."""
    original_text: str
    dialect_text: str
    dialect_name: str
    dimensions: Dict[str, DimensionScore] = field(default_factory=dict)
    overall_score: float = 0.0
    summary: str = ""
    raw_response: str = ""

    def get_score(self, dimension: str) -> Optional[int]:
        """Get score for a specific dimension."""
        if dimension in self.dimensions:
            return self.dimensions[dimension].score
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        dimensions_dict = {}
        for k, v in self.dimensions.items():
            dim_data = {
                "score": v.score,
                "reasoning": v.reasoning,
                "strengths": v.strengths,
                "weaknesses": v.weaknesses
            }
            # Include feature accuracy specific fields if present
            if v.features_present:
                dim_data["features_present"] = v.features_present
            if v.features_missing:
                dim_data["features_missing"] = v.features_missing
            dimensions_dict[k] = dim_data

        return {
            "original_text": self.original_text,
            "dialect_text": self.dialect_text,
            "dialect_name": self.dialect_name,
            "dimensions": dimensions_dict,
            "overall_score": self.overall_score,
            "summary": self.summary
        }


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

FLUENCY_PROMPT = """You are an expert linguist specializing in English dialect variation, capable of detailed chain-of-thought reasoning.

You are given two pieces of text:
1) Original Text (SAE) – the Standard American English version.
2) Dialect Text – a translated or adapted version in the {dialect} dialect.

Please evaluate the Dialect Text for FLUENCY in {dialect}:

FLUENCY CRITERIA:
- Grammar: Does it follow the grammatical patterns of {dialect}?
- Syntax: Is the word order and sentence structure natural for {dialect}?
- Word Choice: Are vocabulary and expressions authentic to {dialect}?
- Naturalness: Would a native speaker of {dialect} find this natural?
- Consistency: Are dialect features applied consistently throughout?
- Flow: Does the text read smoothly without awkward transitions?

SCORING RUBRIC (1-7):
1: Completely unnatural, pervasive errors, nearly unintelligible in {dialect}
2: Major issues in accuracy/naturalness, very awkward for {dialect} speakers
3: Noticeable errors or unnatural phrasing, only partial alignment with {dialect}
4: Average fluency, some issues but mostly understandable in {dialect}
5: Good fluency, minor errors; generally consistent with {dialect} patterns
6: Very good fluency, rare issues; flows smoothly and naturally in {dialect}
7: Excellent fluency, fully natural, error-free, perfectly aligned with {dialect}

ORIGINAL TEXT (SAE):
{original_text}

DIALECT TEXT ({dialect}):
{dialect_text}

INSTRUCTIONS:
1. Provide a detailed chain-of-thought analysis examining grammar, syntax, word choice, and naturalness
2. Identify specific strengths (what works well)
3. Identify specific weaknesses (what could be improved)
4. End with exactly this format:
   Strengths: [list strengths]
   Weaknesses: [list weaknesses]
   Fluency Score: X (where X is an integer 1-7)

Begin your detailed analysis now."""


FAITHFULNESS_PROMPT = """You are an expert linguist specializing in meaning preservation across dialect transformations.

You are given two pieces of text:
1) Original Text (SAE) – the Standard American English version.
2) Dialect Text – a translated or adapted version in the {dialect} dialect.

Please evaluate the Dialect Text for FAITHFULNESS (meaning preservation):

FAITHFULNESS CRITERIA:
- Semantic Accuracy: Is the core meaning preserved?
- Information Completeness: Are all key facts and details retained?
- Intent Preservation: Is the original purpose/tone maintained?
- No Hallucinations: Does it avoid adding information not in the original?
- No Omissions: Are there missing elements that change the meaning?
- Nuance Retention: Are subtle meanings and implications preserved?

SCORING RUBRIC (1-7):
1: Meaning completely lost or contradicts original
2: Major meaning changes, significant information loss
3: Noticeable meaning shifts, some important details altered
4: Core meaning preserved but with some inaccuracies or omissions
5: Good preservation, minor meaning variations acceptable for dialect
6: Very good preservation, meaning intact with minimal changes
7: Perfect preservation, all meaning, nuance, and intent retained

ORIGINAL TEXT (SAE):
{original_text}

DIALECT TEXT ({dialect}):
{dialect_text}

INSTRUCTIONS:
1. Compare the texts carefully, identifying what meaning is preserved and what is changed
2. Note any additions, omissions, or alterations
3. Consider whether changes are acceptable dialect adaptations vs. meaning errors
4. End with exactly this format:
   Strengths: [list strengths]
   Weaknesses: [list weaknesses]
   Faithfulness Score: X (where X is an integer 1-7)

Begin your detailed analysis now."""


DIALECT_AUTHENTICITY_PROMPT = """You are an expert sociolinguist specializing in {dialect} and its distinctive features.

You are given two pieces of text:
1) Original Text (SAE) – the Standard American English version.
2) Dialect Text – a translated or adapted version in the {dialect} dialect.

Please evaluate the Dialect Text for DIALECT AUTHENTICITY:

AUTHENTICITY CRITERIA:
- Feature Accuracy: Are the dialect features correctly applied?
- Feature Completeness: Are appropriate features used where applicable?
- No Overcorrection: Are features applied appropriately (not forced)?
- Register Consistency: Is the formality level consistent?
- Cultural Alignment: Does it reflect the cultural context of {dialect}?
- Avoids Stereotypes: Does it represent the dialect respectfully and accurately?

KNOWN FEATURES OF {dialect} TO CHECK:
- Grammatical patterns (verb forms, negation, agreement)
- Pronoun usage and forms
- Tense and aspect markers
- Vocabulary and expressions
- Discourse markers and particles

SCORING RUBRIC (1-7):
1: No authentic dialect features, reads as SAE or incorrect dialect
2: Few features present, mostly inauthentic or incorrectly applied
3: Some authentic features but inconsistent or partially incorrect
4: Moderate authenticity, recognizable as {dialect} with some issues
5: Good authenticity, most features correct and appropriate
6: Very good authenticity, accurate and consistent feature use
7: Excellent authenticity, indistinguishable from native {dialect} speaker

ORIGINAL TEXT (SAE):
{original_text}

DIALECT TEXT ({dialect}):
{dialect_text}

INSTRUCTIONS:
1. Identify specific dialect features present in the text
2. Evaluate whether each feature is correctly applied
3. Note any missing opportunities to apply appropriate features
4. Check for any stereotypical or inauthentic elements
5. End with exactly this format:
   Strengths: [list strengths]
   Weaknesses: [list weaknesses]
   Dialect Authenticity Score: X (where X is an integer 1-7)

Begin your detailed analysis now."""


FEATURE_ACCURACY_PROMPT = """You are an expert dialectologist with deep knowledge of the eWAVE (Electronic World Atlas of Varieties of English) feature specifications.

You are given two pieces of text:
1) Original Text (SAE) – the Standard American English version.
2) Dialect Text – a translated or adapted version in the {dialect} dialect.

Please evaluate the Dialect Text for FEATURE ACCURACY based on eWAVE linguistic specifications:

eWAVE FEATURE CATEGORIES TO CHECK:
═══════════════════════════════════════════════════════════════

1. PRONOUNS (eWAVE 1-47):
   - Pronoun exchange (she/he for inanimate, me for I in coordinates)
   - Regularized reflexives (hisself, theirselves)
   - Second person plural forms (y'all, youse, you guys)
   - Object/possessive distinctions

2. NOUN PHRASE (eWAVE 48-87):
   - Double/multiple determiners (them books, this here)
   - Zero/regularized plurals
   - Associative plurals (and them, dem)

3. TENSE & ASPECT (eWAVE 88-113):
   - Habitual markers (invariant be, does be)
   - Completive markers (done, been)
   - Anterior markers (had + past, been + V-ing)
   - Progressive extensions

4. MODAL VERBS (eWAVE 114-127):
   - Double modals (might could, might would)
   - Quasi-modals (finna, gonna, liketa)
   - Epistemic variants

5. VERB MORPHOLOGY (eWAVE 128-153):
   - A-prefixing (a-going, a-hunting)
   - Unmarked past tense
   - Leveled verb forms (give for gave, seen for saw)

6. NEGATION (eWAVE 154-169):
   - Ain't usage
   - Negative concord (double/triple negatives)
   - Never as past tense negator
   - Negative tags

7. AGREEMENT (eWAVE 170-184):
   - Copula deletion
   - Zero 3sg -s (he go, she work)
   - Was/were leveling
   - Existential constructions (there's/it's + plural)

8. RELATIVIZATION (eWAVE 185-199):
   - Zero relative markers
   - What/that/as relativizers
   - Resumptive pronouns

9. COMPLEMENTATION (eWAVE 200-210):
   - Say complementizer
   - For-to infinitives
   - Bare infinitive complements

10. ADVERBIAL SUBORDINATION (eWAVE 211-215):
    - Clause-final particles (but, though)
    - Conjunction patterns

11. ADVERBS & PREPOSITIONS (eWAVE 216-222):
    - Flat adverbs (real good, come quick)
    - Preposition variation

12. DISCOURSE & WORD ORDER (eWAVE 223-235):
    - Negative inversion
    - Topicalization patterns
    - Discourse markers

FEATURE ACCURACY CRITERIA:
- Feature Presence: Are the expected eWAVE features for {dialect} present?
- Feature Correctness: Are features applied according to eWAVE specifications?
- Feature Context: Are features used in appropriate grammatical contexts?
- Feature Consistency: Are features applied consistently throughout?
- Feature Coverage: Are features applied wherever applicable in the text?
- Avoid Hypercorrection: Are features natural, not forced or over-applied?

SCORING RUBRIC (1-7):
1: No eWAVE features present, or features grossly incorrect
2: Few features present, mostly incorrectly applied per eWAVE specs
3: Some features present but inconsistent with eWAVE specifications
4: Moderate feature accuracy, some alignment with eWAVE but gaps exist
5: Good feature accuracy, most features align with eWAVE specifications
6: Very good accuracy, features correctly applied per eWAVE with minor issues
7: Excellent accuracy, full alignment with eWAVE specifications for {dialect}

ORIGINAL TEXT (SAE):
{original_text}

DIALECT TEXT ({dialect}):
{dialect_text}

INSTRUCTIONS:
1. Identify which eWAVE features SHOULD be present based on the input and target dialect
2. Check whether those features are correctly applied in the dialect text
3. Note any missing features that should have been applied
4. Identify any incorrectly applied features
5. Consider feature coverage across all applicable contexts
6. End with exactly this format:
   Features Present: [list eWAVE features identified]
   Features Missing: [list expected but absent features]
   Features Incorrect: [list incorrectly applied features]
   Strengths: [list strengths]
   Weaknesses: [list weaknesses]
   Feature Accuracy Score: X (where X is an integer 1-7)

Begin your detailed analysis now."""


COHERENCE_PROMPT = """You are an expert in discourse analysis and text coherence.

You are given a Dialect Text that has been transformed from Standard American English to {dialect}.

Please evaluate the Dialect Text for COHERENCE:

COHERENCE CRITERIA:
- Logical Flow: Do ideas connect logically from one to the next?
- Referential Clarity: Are pronouns and references clear?
- Topic Continuity: Is the topic maintained consistently?
- Transitions: Are shifts between ideas smooth?
- Internal Consistency: Are there contradictions or confusing elements?
- Discourse Structure: Is the overall organization maintained?

SCORING RUBRIC (1-7):
1: Incoherent, no logical connection between parts
2: Severely disjointed, very difficult to follow
3: Noticeable coherence issues, reader must work to understand
4: Moderately coherent, some rough transitions or unclear references
5: Good coherence, generally flows well with minor issues
6: Very good coherence, smooth and easy to follow
7: Excellent coherence, perfectly organized and crystal clear

DIALECT TEXT ({dialect}):
{dialect_text}

INSTRUCTIONS:
1. Read through the text analyzing how ideas connect
2. Identify any breaks in logic or unclear references
3. Evaluate the overall flow and structure
4. End with exactly this format:
   Strengths: [list strengths]
   Weaknesses: [list weaknesses]
   Coherence Score: X (where X is an integer 1-7)

Begin your detailed analysis now."""


READABILITY_PROMPT = """You are an expert in text accessibility and readability assessment.

You are given a Dialect Text in {dialect}.

Please evaluate the Dialect Text for READABILITY:

READABILITY CRITERIA:
- Comprehensibility: Is the text easy to understand?
- Sentence Complexity: Are sentences appropriately structured?
- Word Accessibility: Are words and expressions understandable?
- Cognitive Load: Does reading require excessive effort?
- Clarity: Is the message clear without ambiguity?
- Audience Appropriateness: Would the intended audience understand this?

Consider that this is a {dialect} text, so some features may differ from SAE while still being readable to speakers of that variety.

SCORING RUBRIC (1-7):
1: Unreadable, cannot be understood
2: Very difficult to read, requires significant effort
3: Challenging to read, multiple re-readings needed
4: Moderately readable, some effort required
5: Good readability, generally easy to understand
6: Very good readability, flows smoothly
7: Excellent readability, effortlessly comprehensible

DIALECT TEXT ({dialect}):
{dialect_text}

INSTRUCTIONS:
1. Read through the text as a potential audience member would
2. Note any points of confusion or difficulty
3. Consider both dialect speakers and general English readers
4. End with exactly this format:
   Strengths: [list strengths]
   Weaknesses: [list weaknesses]
   Readability Score: X (where X is an integer 1-7)

Begin your detailed analysis now."""


COMPREHENSIVE_PROMPT = """You are an expert linguist and evaluator specializing in dialect transformations.

You are given two pieces of text:
1) Original Text (SAE) – the Standard American English version.
2) Dialect Text – a translated or adapted version in the {dialect} dialect.

Please provide a COMPREHENSIVE EVALUATION across all dimensions:

═══════════════════════════════════════════════════════════════
EVALUATION DIMENSIONS (Rate each 1-7):
═══════════════════════════════════════════════════════════════

1. FLUENCY: Grammar, syntax, word choice, naturalness in {dialect}
2. FAITHFULNESS: Meaning preservation from the original
3. DIALECT AUTHENTICITY: Accuracy and completeness of dialect features
4. FEATURE ACCURACY: Alignment with eWAVE linguistic feature specifications
5. COHERENCE: Logical flow, referential clarity, organization
6. READABILITY: Ease of comprehension for the target audience

eWAVE FEATURE CATEGORIES (for Feature Accuracy evaluation):
- Pronouns (1-47): pronoun exchange, reflexives, 2nd person plural
- Noun Phrase (48-87): determiners, plurals
- Tense & Aspect (88-113): habitual be, completive done/been
- Modal Verbs (114-127): double modals, quasi-modals
- Verb Morphology (128-153): a-prefixing, leveled forms
- Negation (154-169): ain't, negative concord
- Agreement (170-184): copula deletion, zero 3sg -s
- Relativization (185-199): zero relatives, resumptive pronouns
- Complementation (200-210): say complementizer
- Adverbial Subordination (211-215): clause-final particles
- Adverbs & Prepositions (216-222): flat adverbs
- Discourse & Word Order (223-235): negative inversion, topicalization

SCORING RUBRIC (applies to all dimensions):
1: Completely inadequate
2: Major issues
3: Noticeable problems
4: Average/acceptable
5: Good
6: Very good
7: Excellent

═══════════════════════════════════════════════════════════════
TEXTS TO EVALUATE:
═══════════════════════════════════════════════════════════════

ORIGINAL TEXT (SAE):
{original_text}

DIALECT TEXT ({dialect}):
{dialect_text}

═══════════════════════════════════════════════════════════════
INSTRUCTIONS:
═══════════════════════════════════════════════════════════════

Provide your evaluation in the following JSON format:

```json
{{
    "fluency": {{
        "score": <1-7>,
        "reasoning": "<explanation>",
        "strengths": ["<strength1>", "<strength2>"],
        "weaknesses": ["<weakness1>", "<weakness2>"]
    }},
    "faithfulness": {{
        "score": <1-7>,
        "reasoning": "<explanation>",
        "strengths": ["<strength1>"],
        "weaknesses": ["<weakness1>"]
    }},
    "dialect_authenticity": {{
        "score": <1-7>,
        "reasoning": "<explanation>",
        "strengths": ["<strength1>"],
        "weaknesses": ["<weakness1>"]
    }},
    "feature_accuracy": {{
        "score": <1-7>,
        "reasoning": "<explanation of eWAVE feature alignment>",
        "strengths": ["<eWAVE features correctly applied>"],
        "weaknesses": ["<missing or incorrect eWAVE features>"],
        "features_present": ["<list of eWAVE features identified>"],
        "features_missing": ["<expected but absent features>"]
    }},
    "coherence": {{
        "score": <1-7>,
        "reasoning": "<explanation>",
        "strengths": ["<strength1>"],
        "weaknesses": ["<weakness1>"]
    }},
    "readability": {{
        "score": <1-7>,
        "reasoning": "<explanation>",
        "strengths": ["<strength1>"],
        "weaknesses": ["<weakness1>"]
    }},
    "overall_score": <average of all scores, rounded to 1 decimal>,
    "summary": "<2-3 sentence overall assessment>"
}}
```

Provide ONLY the JSON output, no additional text."""


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class DialectEvaluator:
    """
    LLM-as-a-Judge evaluator for dialect transformations.

    Supports multiple LLM backends: OpenAI, Anthropic, Ollama, HuggingFace.
    Evaluates across 6 dimensions: fluency, faithfulness, dialect authenticity,
    feature accuracy (eWAVE alignment), coherence, and readability.
    """

    DIMENSION_PROMPTS = {
        "fluency": FLUENCY_PROMPT,
        "faithfulness": FAITHFULNESS_PROMPT,
        "dialect_authenticity": DIALECT_AUTHENTICITY_PROMPT,
        "feature_accuracy": FEATURE_ACCURACY_PROMPT,
        "coherence": COHERENCE_PROMPT,
        "readability": READABILITY_PROMPT,
    }

    def __init__(self, llm=None, verbose: bool = False):
        """
        Initialize the evaluator.

        Args:
            llm: LLM backend instance (OpenAI, Ollama, HuggingFace, etc.)
            verbose: Whether to print detailed output
        """
        self.llm = llm
        self.verbose = verbose

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        if self.llm is None:
            raise ValueError("LLM backend required. Pass llm= to constructor.")

        # Use the BaseLLM interface with system and user prompts
        response = self.llm.generate(
            system="You are an expert linguist and dialectologist evaluating dialect transformations.",
            user=prompt
        )
        return response

    def _parse_score(self, response: str, dimension: str) -> DimensionScore:
        """Parse a single dimension score from LLM response."""
        # Extract score
        score_pattern = rf'{dimension.replace("_", " ").title()} Score:\s*(\d)'
        score_match = re.search(score_pattern, response, re.IGNORECASE)

        if not score_match:
            # Try simpler pattern
            score_match = re.search(r'Score:\s*(\d)', response)

        score = int(score_match.group(1)) if score_match else 4

        # Extract strengths
        strengths = []
        strengths_match = re.search(r'Strengths?:\s*\[([^\]]+)\]', response, re.IGNORECASE)
        if strengths_match:
            strengths = [s.strip().strip('"\'') for s in strengths_match.group(1).split(',')]
        else:
            # Try bullet point format
            strengths_section = re.search(r'Strengths?:(.*?)(?:Weaknesses?:|Score:|$)', response, re.IGNORECASE | re.DOTALL)
            if strengths_section:
                strengths = re.findall(r'[-•]\s*(.+)', strengths_section.group(1))

        # Extract weaknesses
        weaknesses = []
        weaknesses_match = re.search(r'Weaknesses?:\s*\[([^\]]+)\]', response, re.IGNORECASE)
        if weaknesses_match:
            weaknesses = [w.strip().strip('"\'') for w in weaknesses_match.group(1).split(',')]
        else:
            weaknesses_section = re.search(r'Weaknesses?:(.*?)(?:Score:|$)', response, re.IGNORECASE | re.DOTALL)
            if weaknesses_section:
                weaknesses = re.findall(r'[-•]\s*(.+)', weaknesses_section.group(1))

        # Extract feature accuracy specific fields (for feature_accuracy dimension)
        features_present = []
        features_missing = []
        if dimension == "feature_accuracy":
            # Extract features present
            present_match = re.search(r'Features?\s*Present:\s*\[([^\]]+)\]', response, re.IGNORECASE)
            if present_match:
                features_present = [f.strip().strip('"\'') for f in present_match.group(1).split(',')]
            else:
                present_section = re.search(r'Features?\s*Present:(.*?)(?:Features?\s*Missing:|Strengths?:|$)', response, re.IGNORECASE | re.DOTALL)
                if present_section:
                    features_present = re.findall(r'[-•]\s*(.+)', present_section.group(1))

            # Extract features missing
            missing_match = re.search(r'Features?\s*Missing:\s*\[([^\]]+)\]', response, re.IGNORECASE)
            if missing_match:
                features_missing = [f.strip().strip('"\'') for f in missing_match.group(1).split(',')]
            else:
                missing_section = re.search(r'Features?\s*Missing:(.*?)(?:Features?\s*Incorrect:|Strengths?:|$)', response, re.IGNORECASE | re.DOTALL)
                if missing_section:
                    features_missing = re.findall(r'[-•]\s*(.+)', missing_section.group(1))

        return DimensionScore(
            dimension=dimension,
            score=score,
            reasoning=response[:500] + "..." if len(response) > 500 else response,
            strengths=strengths[:5],  # Limit to 5
            weaknesses=weaknesses[:5],
            features_present=features_present[:10],  # Limit to 10
            features_missing=features_missing[:10]
        )

    def _parse_comprehensive_response(self, response: str) -> Dict:
        """Parse the comprehensive JSON response."""
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Return default structure
            return {
                "fluency": {"score": 4, "reasoning": "Could not parse response", "strengths": [], "weaknesses": []},
                "faithfulness": {"score": 4, "reasoning": "Could not parse response", "strengths": [], "weaknesses": []},
                "dialect_authenticity": {"score": 4, "reasoning": "Could not parse response", "strengths": [], "weaknesses": []},
                "feature_accuracy": {"score": 4, "reasoning": "Could not parse response", "strengths": [], "weaknesses": [], "features_present": [], "features_missing": []},
                "coherence": {"score": 4, "reasoning": "Could not parse response", "strengths": [], "weaknesses": []},
                "readability": {"score": 4, "reasoning": "Could not parse response", "strengths": [], "weaknesses": []},
                "overall_score": 4.0,
                "summary": "Evaluation parsing failed. Please check raw response."
            }

    def evaluate_dimension(
        self,
        original_text: str,
        dialect_text: str,
        dialect_name: str,
        dimension: str
    ) -> DimensionScore:
        """
        Evaluate a single dimension.

        Args:
            original_text: Original SAE text
            dialect_text: Transformed dialect text
            dialect_name: Name of the target dialect
            dimension: Dimension to evaluate (fluency, faithfulness, etc.)

        Returns:
            DimensionScore with score and reasoning
        """
        if dimension not in self.DIMENSION_PROMPTS:
            raise ValueError(f"Unknown dimension: {dimension}. Available: {list(self.DIMENSION_PROMPTS.keys())}")

        prompt_template = self.DIMENSION_PROMPTS[dimension]
        prompt = prompt_template.format(
            dialect=dialect_name,
            original_text=original_text,
            dialect_text=dialect_text
        )

        if self.verbose:
            print(f"\n=== Evaluating {dimension.upper()} ===")

        response = self._call_llm(prompt)

        if self.verbose:
            print(response[:500] + "..." if len(response) > 500 else response)

        return self._parse_score(response, dimension)

    def evaluate(
        self,
        original_text: str,
        dialect_text: str,
        dialect_name: str,
        dimensions: Optional[List[str]] = None,
        comprehensive: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a dialect transformation across multiple dimensions.

        Args:
            original_text: Original SAE text
            dialect_text: Transformed dialect text
            dialect_name: Name of the target dialect
            dimensions: List of dimensions to evaluate (default: all)
            comprehensive: Use single comprehensive prompt (faster) vs individual prompts

        Returns:
            EvaluationResult with all scores and analysis
        """
        if dimensions is None:
            dimensions = list(self.DIMENSION_PROMPTS.keys())

        result = EvaluationResult(
            original_text=original_text,
            dialect_text=dialect_text,
            dialect_name=dialect_name
        )

        if comprehensive:
            # Single comprehensive evaluation
            prompt = COMPREHENSIVE_PROMPT.format(
                dialect=dialect_name,
                original_text=original_text,
                dialect_text=dialect_text
            )

            if self.verbose:
                print("\n=== Comprehensive Evaluation ===")

            response = self._call_llm(prompt)
            result.raw_response = response

            parsed = self._parse_comprehensive_response(response)

            for dim in dimensions:
                if dim in parsed:
                    dim_data = parsed[dim]
                    result.dimensions[dim] = DimensionScore(
                        dimension=dim,
                        score=dim_data.get("score", 4),
                        reasoning=dim_data.get("reasoning", ""),
                        strengths=dim_data.get("strengths", []),
                        weaknesses=dim_data.get("weaknesses", []),
                        features_present=dim_data.get("features_present", []),
                        features_missing=dim_data.get("features_missing", [])
                    )

            result.overall_score = parsed.get("overall_score", 0)
            result.summary = parsed.get("summary", "")

            if result.overall_score == 0 and result.dimensions:
                result.overall_score = round(
                    sum(d.score for d in result.dimensions.values()) / len(result.dimensions),
                    1
                )

        else:
            # Individual dimension evaluations
            for dim in dimensions:
                score = self.evaluate_dimension(original_text, dialect_text, dialect_name, dim)
                result.dimensions[dim] = score

            # Calculate overall score
            if result.dimensions:
                result.overall_score = round(
                    sum(d.score for d in result.dimensions.values()) / len(result.dimensions),
                    1
                )

            # Generate summary
            result.summary = self._generate_summary(result)

        if self.verbose:
            print(f"\n=== Evaluation Complete ===")
            print(f"Overall Score: {result.overall_score}/7")
            for dim, score in result.dimensions.items():
                print(f"  {dim}: {score.score}/7")

        return result

    def _generate_summary(self, result: EvaluationResult) -> str:
        """Generate a summary from individual dimension scores."""
        if not result.dimensions:
            return "No dimensions evaluated."

        scores = {dim: s.score for dim, s in result.dimensions.items()}
        highest = max(scores, key=scores.get)
        lowest = min(scores, key=scores.get)

        return (
            f"The {result.dialect_name} transformation scored {result.overall_score}/7 overall. "
            f"Strongest in {highest.replace('_', ' ')} ({scores[highest]}/7), "
            f"needs improvement in {lowest.replace('_', ' ')} ({scores[lowest]}/7)."
        )

    def evaluate_batch(
        self,
        pairs: List[Dict[str, str]],
        dialect_name: str,
        comprehensive: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple text pairs.

        Args:
            pairs: List of {"original": str, "dialect": str} dictionaries
            dialect_name: Name of the target dialect
            comprehensive: Use comprehensive prompt

        Returns:
            List of EvaluationResult objects
        """
        results = []
        for i, pair in enumerate(pairs):
            if self.verbose:
                print(f"\n=== Evaluating pair {i+1}/{len(pairs)} ===")

            result = self.evaluate(
                original_text=pair["original"],
                dialect_text=pair["dialect"],
                dialect_name=dialect_name,
                comprehensive=comprehensive
            )
            results.append(result)

        return results


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_transformation(
    original_text: str,
    dialect_text: str,
    dialect_name: str,
    llm=None,
    verbose: bool = False
) -> EvaluationResult:
    """
    Convenience function to evaluate a single transformation.

    Args:
        original_text: Original SAE text
        dialect_text: Transformed dialect text
        dialect_name: Name of the target dialect
        llm: LLM backend instance
        verbose: Print detailed output

    Returns:
        EvaluationResult with scores and analysis
    """
    evaluator = DialectEvaluator(llm=llm, verbose=verbose)
    return evaluator.evaluate(original_text, dialect_text, dialect_name)


def quick_score(
    original_text: str,
    dialect_text: str,
    dialect_name: str,
    llm=None
) -> float:
    """
    Get just the overall score quickly.

    Args:
        original_text: Original SAE text
        dialect_text: Transformed dialect text
        dialect_name: Name of the target dialect
        llm: LLM backend instance

    Returns:
        Overall score (1-7)
    """
    result = evaluate_transformation(
        original_text, dialect_text, dialect_name, llm=llm, verbose=False
    )
    return result.overall_score


def build_evaluation_prompt(
    original_text: str,
    dialect_text: str,
    dialect_name: str,
    dimension: str = "comprehensive"
) -> str:
    """
    Build an evaluation prompt for custom LLM integration.

    Args:
        original_text: Original SAE text
        dialect_text: Transformed dialect text
        dialect_name: Name of the target dialect
        dimension: Which dimension to evaluate (or "comprehensive" for all)

    Returns:
        Formatted prompt string
    """
    if dimension == "comprehensive":
        return COMPREHENSIVE_PROMPT.format(
            dialect=dialect_name,
            original_text=original_text,
            dialect_text=dialect_text
        )

    if dimension not in DialectEvaluator.DIMENSION_PROMPTS:
        raise ValueError(f"Unknown dimension: {dimension}")

    return DialectEvaluator.DIMENSION_PROMPTS[dimension].format(
        dialect=dialect_name,
        original_text=original_text,
        dialect_text=dialect_text
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(results: List[EvaluationResult]) -> Dict:
    """
    Aggregate evaluation results across multiple texts.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Dictionary with aggregated statistics
    """
    if not results:
        return {"error": "No results to aggregate"}

    # Collect all dimension scores
    dimension_scores = {}
    for result in results:
        for dim, score in result.dimensions.items():
            if dim not in dimension_scores:
                dimension_scores[dim] = []
            dimension_scores[dim].append(score.score)

    # Calculate statistics
    stats = {
        "total_samples": len(results),
        "dimensions": {},
        "overall": {
            "mean": round(sum(r.overall_score for r in results) / len(results), 2),
            "min": min(r.overall_score for r in results),
            "max": max(r.overall_score for r in results)
        }
    }

    for dim, scores in dimension_scores.items():
        stats["dimensions"][dim] = {
            "mean": round(sum(scores) / len(scores), 2),
            "min": min(scores),
            "max": max(scores),
            "distribution": {i: scores.count(i) for i in range(1, 8) if scores.count(i) > 0}
        }

    return stats


def format_results_report(results: List[EvaluationResult]) -> str:
    """
    Format evaluation results as a readable report.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Formatted report string
    """
    stats = aggregate_results(results)

    report = []
    report.append("=" * 60)
    report.append("DIALECT TRANSFORMATION EVALUATION REPORT")
    report.append("=" * 60)
    report.append(f"\nTotal Samples Evaluated: {stats['total_samples']}")
    report.append(f"Overall Mean Score: {stats['overall']['mean']}/7")
    report.append(f"Score Range: {stats['overall']['min']} - {stats['overall']['max']}")

    report.append("\n" + "-" * 60)
    report.append("DIMENSION BREAKDOWN:")
    report.append("-" * 60)

    for dim, dim_stats in stats.get("dimensions", {}).items():
        report.append(f"\n{dim.upper().replace('_', ' ')}:")
        report.append(f"  Mean: {dim_stats['mean']}/7")
        report.append(f"  Range: {dim_stats['min']} - {dim_stats['max']}")
        if dim_stats.get("distribution"):
            dist_str = ", ".join([f"{k}:{v}" for k, v in sorted(dim_stats["distribution"].items())])
            report.append(f"  Distribution: {dist_str}")

    report.append("\n" + "=" * 60)

    return "\n".join(report)
