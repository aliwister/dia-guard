# coi_reintegration.py - Chain of Interaction (CoI) for Sensitive Token Reintegration
#
# Handles reintegration of sensitive tokens (emoji, URLs, special characters) that were
# removed before Multi-VALUE processing due to OOV errors.
#
# Based on AXL-CoI framework: https://aclanthology.org/2025.findings-emnlp.191.pdf
# Adapted for dialect transformation with dual attention and self-corrective generation.

import re
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SensitiveToken:
    """Represents a sensitive token that was removed from original text."""
    token: str
    token_type: str  # emoji, url, mention, hashtag, special_char, code_block
    position: int  # Original position in text
    context_before: str  # Text before token (for context matching)
    context_after: str  # Text after token (for context matching)


@dataclass
class ChainOutput:
    """Output from a single chain in the CoI process."""
    chain_id: int
    chain_name: str
    output: Any
    reasoning: Optional[str] = None


@dataclass
class CoIResult:
    """Final result from the Chain of Interaction process."""
    original_text: str
    multivalue_text: str
    final_output: str
    chain_outputs: List[ChainOutput] = field(default_factory=list)
    tokens_reintegrated: List[SensitiveToken] = field(default_factory=list)
    validation_passed: bool = True
    corrections_made: List[str] = field(default_factory=list)


# Patterns for detecting sensitive tokens
SENSITIVE_TOKEN_PATTERNS = {
    "emoji": re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002600-\U000026FF"  # misc symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "]+",
        re.UNICODE
    ),
    "url": re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+|'
        r'www\.[^\s<>"{}|\\^`\[\]]+'
    ),
    "mention": re.compile(r'@[\w]+'),
    "hashtag": re.compile(r'#[\w]+'),
    "email": re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'),
    "code_block": re.compile(r'```[\s\S]*?```|`[^`\n]+`'),
    "special_char": re.compile(r'[—–―‐‑‒\u2014\u2013\u2012\u2011\u2010]'),  # various dashes
    "file_path": re.compile(r'(?:/[\w.-]+)+|(?:[A-Za-z]:\\[\w\\.-]+)'),
}


class CoIDialectReintegrator:
    """
    Chain of Interaction (CoI) system for reintegrating sensitive tokens
    into Multi-VALUE dialect transformations.

    Uses a 4-chain approach:
    1. Extractor: Extract key details including sensitive tokens from original
    2. Integrator: Add missing items (emoji, URLs, etc.) to dialect text
    3. Validator: Validate changes and check for issues
    4. Corrector: Produce final corrected dialect transformation
    """

    def __init__(self, llm=None, verbose: bool = False):
        """
        Initialize the CoI Reintegrator.

        Args:
            llm: LLM backend for chain processing (optional for rule-based mode)
            verbose: Whether to print chain outputs
        """
        self.llm = llm
        self.verbose = verbose

    def extract_sensitive_tokens(self, text: str) -> List[SensitiveToken]:
        """
        Extract all sensitive tokens from text with their context.

        Args:
            text: Original text with sensitive tokens

        Returns:
            List of SensitiveToken objects
        """
        tokens = []

        for token_type, pattern in SENSITIVE_TOKEN_PATTERNS.items():
            for match in pattern.finditer(text):
                start, end = match.span()

                # Get context (up to 30 chars before/after)
                context_start = max(0, start - 30)
                context_end = min(len(text), end + 30)

                token = SensitiveToken(
                    token=match.group(),
                    token_type=token_type,
                    position=start,
                    context_before=text[context_start:start].strip(),
                    context_after=text[end:context_end].strip()
                )
                tokens.append(token)

        # Sort by position
        tokens.sort(key=lambda t: t.position)
        return tokens

    def find_missing_tokens(
        self,
        original_tokens: List[SensitiveToken],
        target_text: str
    ) -> List[SensitiveToken]:
        """
        Find tokens from original that are missing in target text.

        Args:
            original_tokens: Tokens extracted from original text
            target_text: Multi-VALUE transformed text

        Returns:
            List of missing SensitiveToken objects
        """
        missing = []
        for token in original_tokens:
            if token.token not in target_text:
                missing.append(token)
        return missing

    def _build_chain1_prompt(
        self,
        original_text: str,
        multivalue_text: str,
        sensitive_tokens: List[SensitiveToken]
    ) -> str:
        """Build prompt for Chain 1: Extractor/Analyzer."""
        token_list = "\n".join([
            f"  - {t.token} ({t.token_type}) near: '...{t.context_before}' | '{t.context_after}...'"
            for t in sensitive_tokens
        ])

        return f"""You are an Extractor/Analyzer agent specializing in dialect transformation analysis.

TASK: Analyze the original text and Multi-VALUE transformation to extract key details.

ORIGINAL TEXT:
{original_text}

MULTI-VALUE DIALECT TRANSFORMATION:
{multivalue_text}

SENSITIVE TOKENS DETECTED IN ORIGINAL:
{token_list if token_list else "  (none detected)"}

INSTRUCTIONS:
1. Identify which sensitive tokens (emoji, URLs, mentions, hashtags, special characters) are present in the original but missing from the transformation
2. Note the semantic context where each missing token appeared
3. Identify any grammatical or structural changes made by the dialect transformation
4. Document the relationship between removed tokens and their surrounding context

OUTPUT FORMAT (JSON):
{{
    "missing_tokens": [
        {{
            "token": "<token>",
            "type": "<type>",
            "original_context": "<surrounding text in original>",
            "semantic_role": "<what role does this token play - emphasis, link, identity, emotion, etc.>"
        }}
    ],
    "dialect_changes": [
        "<list of key grammatical/structural changes made by Multi-VALUE>"
    ],
    "preservation_notes": "<notes on what was preserved correctly>"
}}"""

    def _build_chain2_prompt(
        self,
        multivalue_text: str,
        chain1_output: Dict,
        missing_tokens: List[SensitiveToken]
    ) -> str:
        """Build prompt for Chain 2: Integrator/Inserter."""
        return f"""You are an Integrator agent specializing in natural token insertion.

TASK: Reintegrate missing sensitive tokens into the dialect transformation naturally.

MULTI-VALUE DIALECT TRANSFORMATION:
{multivalue_text}

CHAIN 1 ANALYSIS:
{json.dumps(chain1_output, indent=2)}

MISSING TOKENS TO REINTEGRATE:
{json.dumps([{{"token": t.token, "type": t.token_type, "context_before": t.context_before, "context_after": t.context_after}} for t in missing_tokens], indent=2)}

INSTRUCTIONS:
1. For each missing token, find the most natural position to insert it in the dialect text
2. Preserve the dialect features (grammar, vocabulary, syntax) - do NOT undo the dialect transformation
3. Ensure tokens are placed in contextually appropriate locations
4. For emoji: place near equivalent emotional content
5. For URLs/mentions/hashtags: place in structurally similar positions
6. For special characters (dashes, etc.): use dialect-appropriate equivalents if needed

OUTPUT FORMAT (JSON):
{{
    "integrated_text": "<the full dialect text with tokens reintegrated>",
    "insertions": [
        {{
            "token": "<token>",
            "position_description": "<where it was inserted and why>",
            "confidence": "<high/medium/low>"
        }}
    ]
}}"""

    def _build_chain3_prompt(
        self,
        original_text: str,
        multivalue_text: str,
        chain2_output: Dict
    ) -> str:
        """Build prompt for Chain 3: Validator/Checker."""
        return f"""You are a Validator agent specializing in quality assurance for dialect transformations.

TASK: Validate the reintegration of sensitive tokens and check for issues.

ORIGINAL TEXT:
{original_text}

MULTI-VALUE DIALECT TRANSFORMATION (before reintegration):
{multivalue_text}

CHAIN 2 OUTPUT (after reintegration):
{json.dumps(chain2_output, indent=2)}

VALIDATION CHECKLIST:
1. Are all sensitive tokens from the original now present in the integrated text?
2. Are tokens placed in contextually appropriate positions?
3. Is the dialect transformation preserved (grammar, vocabulary, syntax)?
4. Are there any unnatural insertions or awkward placements?
5. Is the meaning/intent of the original preserved?
6. Are there any tokens that should NOT have been inserted (duplicates, wrong context)?

OUTPUT FORMAT (JSON):
{{
    "validation_passed": <true/false>,
    "issues": [
        {{
            "issue_type": "<missing_token/wrong_position/dialect_broken/duplicate/other>",
            "description": "<what the issue is>",
            "suggested_fix": "<how to fix it>"
        }}
    ],
    "tokens_verified": [
        {{
            "token": "<token>",
            "status": "<correctly_placed/needs_adjustment/missing>",
            "notes": "<any notes>"
        }}
    ],
    "dialect_preservation_score": "<1-5, where 5 is perfectly preserved>",
    "overall_quality_score": "<1-5>"
}}"""

    def _build_chain4_prompt(
        self,
        chain2_output: Dict,
        chain3_output: Dict
    ) -> str:
        """Build prompt for Chain 4: Corrector/Finalizer."""
        return f"""You are a Corrector agent producing the final dialect transformation.

TASK: Apply corrections based on validation feedback and produce the final output.

CHAIN 2 INTEGRATED TEXT:
{chain2_output.get('integrated_text', '')}

CHAIN 3 VALIDATION REPORT:
{json.dumps(chain3_output, indent=2)}

INSTRUCTIONS:
1. If validation passed with no issues, return the integrated text as-is
2. If there are issues, apply the suggested fixes
3. Ensure all sensitive tokens are correctly placed
4. Ensure dialect features are fully preserved
5. The output should be natural-sounding in the target dialect

OUTPUT: Return ONLY the final corrected dialect text. No JSON, no explanation, just the text."""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt and return the response."""
        if self.llm is None:
            raise ValueError("LLM backend required for chain processing. Pass llm= to constructor.")

        # Use the BaseLLM interface with system and user prompts
        response = self.llm.generate(
            system="You are an expert at reintegrating sensitive tokens into transformed text.",
            user=prompt
        )
        return response

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
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
            return {"error": "Failed to parse JSON", "raw": response}

    def reintegrate(
        self,
        original_text: str,
        multivalue_text: str,
        use_llm: bool = True
    ) -> CoIResult:
        """
        Run the 4-chain CoI process to reintegrate sensitive tokens.

        Args:
            original_text: Original text with sensitive tokens
            multivalue_text: Multi-VALUE transformed text (tokens removed)
            use_llm: Whether to use LLM for chain processing (False = rule-based)

        Returns:
            CoIResult with final output and chain details
        """
        chain_outputs = []

        # Extract sensitive tokens from original
        original_tokens = self.extract_sensitive_tokens(original_text)
        missing_tokens = self.find_missing_tokens(original_tokens, multivalue_text)

        if self.verbose:
            print(f"Found {len(original_tokens)} sensitive tokens in original")
            print(f"Missing {len(missing_tokens)} tokens in Multi-VALUE output")

        # If no missing tokens, return Multi-VALUE output as-is
        if not missing_tokens:
            return CoIResult(
                original_text=original_text,
                multivalue_text=multivalue_text,
                final_output=multivalue_text,
                chain_outputs=[],
                tokens_reintegrated=[],
                validation_passed=True
            )

        if use_llm and self.llm:
            return self._reintegrate_with_llm(
                original_text, multivalue_text, original_tokens, missing_tokens
            )
        else:
            return self._reintegrate_rule_based(
                original_text, multivalue_text, original_tokens, missing_tokens
            )

    def _reintegrate_with_llm(
        self,
        original_text: str,
        multivalue_text: str,
        original_tokens: List[SensitiveToken],
        missing_tokens: List[SensitiveToken]
    ) -> CoIResult:
        """Run CoI process using LLM for each chain."""
        chain_outputs = []

        # Chain 1: Extractor
        if self.verbose:
            print("\n=== Chain 1: Extractor/Analyzer ===")

        chain1_prompt = self._build_chain1_prompt(original_text, multivalue_text, original_tokens)
        chain1_response = self._call_llm(chain1_prompt)
        chain1_output = self._parse_json_response(chain1_response)

        chain_outputs.append(ChainOutput(
            chain_id=1,
            chain_name="Extractor/Analyzer",
            output=chain1_output
        ))

        if self.verbose:
            print(json.dumps(chain1_output, indent=2))

        # Chain 2: Integrator
        if self.verbose:
            print("\n=== Chain 2: Integrator/Inserter ===")

        chain2_prompt = self._build_chain2_prompt(multivalue_text, chain1_output, missing_tokens)
        chain2_response = self._call_llm(chain2_prompt)
        chain2_output = self._parse_json_response(chain2_response)

        chain_outputs.append(ChainOutput(
            chain_id=2,
            chain_name="Integrator/Inserter",
            output=chain2_output
        ))

        if self.verbose:
            print(json.dumps(chain2_output, indent=2))

        # Chain 3: Validator
        if self.verbose:
            print("\n=== Chain 3: Validator/Checker ===")

        chain3_prompt = self._build_chain3_prompt(original_text, multivalue_text, chain2_output)
        chain3_response = self._call_llm(chain3_prompt)
        chain3_output = self._parse_json_response(chain3_response)

        chain_outputs.append(ChainOutput(
            chain_id=3,
            chain_name="Validator/Checker",
            output=chain3_output
        ))

        if self.verbose:
            print(json.dumps(chain3_output, indent=2))

        # Chain 4: Corrector
        if self.verbose:
            print("\n=== Chain 4: Corrector/Finalizer ===")

        chain4_prompt = self._build_chain4_prompt(chain2_output, chain3_output)
        final_output = self._call_llm(chain4_prompt).strip()

        chain_outputs.append(ChainOutput(
            chain_id=4,
            chain_name="Corrector/Finalizer",
            output=final_output
        ))

        if self.verbose:
            print(f"Final output: {final_output}")

        # Build result
        validation_passed = chain3_output.get("validation_passed", True)
        corrections = [
            issue.get("suggested_fix", "")
            for issue in chain3_output.get("issues", [])
        ]

        return CoIResult(
            original_text=original_text,
            multivalue_text=multivalue_text,
            final_output=final_output,
            chain_outputs=chain_outputs,
            tokens_reintegrated=missing_tokens,
            validation_passed=validation_passed,
            corrections_made=[c for c in corrections if c]
        )

    def _reintegrate_rule_based(
        self,
        original_text: str,
        multivalue_text: str,
        original_tokens: List[SensitiveToken],
        missing_tokens: List[SensitiveToken]
    ) -> CoIResult:
        """
        Run CoI process using rule-based heuristics (no LLM required).
        Useful for fast processing or when LLM is unavailable.
        """
        chain_outputs = []
        integrated_text = multivalue_text
        insertions = []

        # Chain 1: Rule-based extraction (already done)
        chain1_output = {
            "missing_tokens": [
                {
                    "token": t.token,
                    "type": t.token_type,
                    "original_context": f"{t.context_before} | {t.context_after}",
                    "semantic_role": self._infer_semantic_role(t)
                }
                for t in missing_tokens
            ],
            "dialect_changes": ["(rule-based mode - changes not analyzed)"],
            "preservation_notes": "Using rule-based reintegration"
        }
        chain_outputs.append(ChainOutput(
            chain_id=1,
            chain_name="Extractor/Analyzer (Rule-based)",
            output=chain1_output
        ))

        # Chain 2: Rule-based integration
        for token in missing_tokens:
            position, confidence = self._find_best_insertion_point(
                token, original_text, integrated_text
            )

            if position >= 0:
                # Insert token at position
                integrated_text = (
                    integrated_text[:position] +
                    token.token +
                    (" " if not integrated_text[position:].startswith(" ") else "") +
                    integrated_text[position:]
                )
                insertions.append({
                    "token": token.token,
                    "position": position,
                    "confidence": confidence
                })

        chain2_output = {
            "integrated_text": integrated_text,
            "insertions": insertions
        }
        chain_outputs.append(ChainOutput(
            chain_id=2,
            chain_name="Integrator/Inserter (Rule-based)",
            output=chain2_output
        ))

        # Chain 3: Rule-based validation
        validation_issues = []
        for token in missing_tokens:
            if token.token not in integrated_text:
                validation_issues.append({
                    "issue_type": "missing_token",
                    "description": f"Token {token.token} could not be reintegrated",
                    "suggested_fix": "Manual insertion required"
                })

        chain3_output = {
            "validation_passed": len(validation_issues) == 0,
            "issues": validation_issues,
            "tokens_verified": [
                {
                    "token": t.token,
                    "status": "correctly_placed" if t.token in integrated_text else "missing"
                }
                for t in missing_tokens
            ]
        }
        chain_outputs.append(ChainOutput(
            chain_id=3,
            chain_name="Validator/Checker (Rule-based)",
            output=chain3_output
        ))

        # Chain 4: Final output (no corrections in rule-based mode)
        final_output = integrated_text
        chain_outputs.append(ChainOutput(
            chain_id=4,
            chain_name="Corrector/Finalizer (Rule-based)",
            output=final_output
        ))

        return CoIResult(
            original_text=original_text,
            multivalue_text=multivalue_text,
            final_output=final_output,
            chain_outputs=chain_outputs,
            tokens_reintegrated=[t for t in missing_tokens if t.token in final_output],
            validation_passed=chain3_output["validation_passed"],
            corrections_made=[]
        )

    def _infer_semantic_role(self, token: SensitiveToken) -> str:
        """Infer the semantic role of a token based on its type and context."""
        role_map = {
            "emoji": "emotional_expression",
            "url": "reference_link",
            "mention": "social_reference",
            "hashtag": "topic_marker",
            "email": "contact_info",
            "code_block": "technical_content",
            "special_char": "punctuation",
            "file_path": "file_reference"
        }
        return role_map.get(token.token_type, "unknown")

    def _find_best_insertion_point(
        self,
        token: SensitiveToken,
        original_text: str,
        target_text: str
    ) -> Tuple[int, str]:
        """
        Find the best position to insert a token in target text.

        Returns:
            Tuple of (position, confidence)
        """
        # Strategy 1: Look for context matches
        if token.context_before:
            # Find similar context in target
            context_words = token.context_before.split()[-3:]  # Last 3 words
            for i in range(len(context_words), 0, -1):
                search_phrase = " ".join(context_words[-i:])
                if search_phrase in target_text:
                    pos = target_text.find(search_phrase) + len(search_phrase)
                    return pos, "high" if i >= 2 else "medium"

        # Strategy 2: Position-based (proportional)
        if token.position > 0 and len(original_text) > 0:
            ratio = token.position / len(original_text)
            target_pos = int(ratio * len(target_text))
            # Find nearest word boundary
            while target_pos < len(target_text) and target_text[target_pos] not in " \n\t":
                target_pos += 1
            return target_pos, "low"

        # Strategy 3: End of text for emoji, start for URLs
        if token.token_type == "emoji":
            return len(target_text), "low"
        elif token.token_type in ("url", "hashtag", "mention"):
            # Find end of sentence or text
            for punct in [".", "!", "?", "\n"]:
                if punct in target_text:
                    return target_text.find(punct), "low"

        return len(target_text), "low"


# Convenience function for quick reintegration
def reintegrate_tokens(
    original_text: str,
    multivalue_text: str,
    llm=None,
    verbose: bool = False
) -> str:
    """
    Convenience function to reintegrate sensitive tokens.

    Args:
        original_text: Original text with sensitive tokens
        multivalue_text: Multi-VALUE transformed text
        llm: Optional LLM backend
        verbose: Print chain outputs

    Returns:
        Final text with tokens reintegrated
    """
    reintegrator = CoIDialectReintegrator(llm=llm, verbose=verbose)
    result = reintegrator.reintegrate(
        original_text,
        multivalue_text,
        use_llm=(llm is not None)
    )
    return result.final_output


# CoI Prompt Template for direct LLM use
COI_REINTEGRATION_PROMPT = """You are a Dialect Transformation Reintegration System using Chain of Interaction (CoI).

You will receive:
1. ORIGINAL TEXT: The source text containing sensitive tokens (emoji, URLs, mentions, etc.)
2. MULTI-VALUE TRANSFORMATION: The dialect-transformed text with sensitive tokens removed

Your task is to reintegrate the missing sensitive tokens naturally while preserving the dialect features.

Execute the following 4-chain process internally, then output ONLY the final result:

---
CHAIN 1 - EXTRACTOR/ANALYZER:
- Extract all sensitive tokens from original (emoji, URLs, @mentions, #hashtags, special characters)
- Identify which tokens are missing from the transformation
- Note the context where each token appeared

CHAIN 2 - INTEGRATOR/INSERTER:
- For each missing token, find the most natural position in the dialect text
- Insert tokens while preserving dialect grammar and vocabulary
- Place emoji near equivalent emotional content
- Place URLs/mentions/hashtags in structurally similar positions

CHAIN 3 - VALIDATOR/CHECKER:
- Verify all missing tokens are now present
- Check tokens are contextually appropriate
- Confirm dialect features are preserved
- Flag any issues

CHAIN 4 - CORRECTOR/FINALIZER:
- Apply any needed corrections
- Produce the final polished dialect text with all tokens reintegrated
---

ORIGINAL TEXT:
{original_text}

MULTI-VALUE TRANSFORMATION:
{multivalue_text}

OUTPUT: Return ONLY the final corrected dialect transformation with sensitive tokens reintegrated. No explanation, no JSON, just the final text."""


def build_coi_prompt(original_text: str, multivalue_text: str) -> str:
    """Build a CoI prompt for direct LLM use."""
    return COI_REINTEGRATION_PROMPT.format(
        original_text=original_text,
        multivalue_text=multivalue_text
    )
