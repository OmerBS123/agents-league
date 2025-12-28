"""
Ollama integration for LLM-powered game strategy.
Implements timeout handling, response parsing, and fallback strategies.
"""

import asyncio
import json
import re
import time
from typing import List, Dict, Optional, Literal, Any
from dataclasses import dataclass

import httpx

from shared.logger import get_logger, log_with_context, log_performance
from shared.exceptions import LLMError, OperationTimeoutError, NetworkError
from shared.schemas import ParityChoice
from consts import (
    OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, LLM_TIMEOUT, LLM_TEMPERATURE, LLM_MAX_TOKENS
)

logger = get_logger(__name__, "ollama-strategy")


# --- Data Structures ---

@dataclass
class GameContext:
    """Context information for LLM strategy decisions"""
    agent_id: str
    opponent_id: str
    match_id: str
    round_number: int
    opponent_history: List[ParityChoice]
    my_history: List[ParityChoice]
    current_score: Dict[str, int]  # {player_id: score}

    @property
    def score_summary(self) -> str:
        """Human-readable score summary"""
        if not self.current_score or len(self.current_score) != 2:
            return "0-0"

        scores = list(self.current_score.values())
        return f"{scores[0]}-{scores[1]}"


# --- Helper Classes ---

class PromptBuilder:
    """
    Constructs optimized prompts for game theory decision making.
    Uses advanced prompt engineering techniques for better LLM performance.
    """

    @staticmethod
    def build_system_prompt() -> str:
        """System prompt defining the LLM's role and constraints"""
        return (
            "You are an expert Game Theory AI playing the 'Even/Odd' number game. "
            "Your goal is to maximize your wins by making optimal strategic decisions.\n\n"

            "GAME RULES:\n"
            "1. Two players simultaneously choose 'even' or 'odd'\n"
            "2. A random number (1-10) is generated\n"
            "3. The sum of your choice value + opponent choice value + random number determines the winner\n"
            "4. Choice values: 'odd' = 1, 'even' = 0\n"
            "5. If the final sum is even, players who chose 'even' win that round\n"
            "6. If the final sum is odd, players who chose 'odd' win that round\n"
            "7. First to win majority of rounds wins the match\n\n"

            "STRATEGIC CONSIDERATIONS:\n"
            "- Analyze opponent patterns and tendencies\n"
            "- Consider probability distributions of random numbers\n"
            "- Use game theory concepts like Nash equilibrium\n"
            "- Adapt your strategy based on match progress\n"
            "- Consider psychological factors and pattern breaking\n\n"

            "OUTPUT REQUIREMENT:\n"
            "Respond with ONLY a valid JSON object in this exact format:\n"
            "{\n"
            '  "choice": "even",\n'
            '  "confidence": 0.75,\n'
            '  "reasoning": "Brief explanation of your strategic reasoning"\n'
            "}\n\n"

            "CONSTRAINTS:\n"
            "- Choice must be exactly 'even' or 'odd'\n"
            "- Confidence must be a number between 0.0 and 1.0\n"
            "- Reasoning must be concise (under 100 characters)\n"
            "- No additional text outside the JSON object"
        )

    @staticmethod
    def build_user_prompt(context: GameContext) -> str:
        """Build context-specific user prompt"""

        # Format opponent history for analysis
        if context.opponent_history:
            opp_moves = [choice.value for choice in context.opponent_history[-10:]]  # Last 10 moves
            opp_hist_str = " -> ".join(opp_moves)

            # Pattern analysis
            even_count = sum(1 for choice in context.opponent_history if choice == ParityChoice.EVEN)
            odd_count = len(context.opponent_history) - even_count
            opp_tendency = f"({even_count} even, {odd_count} odd)"
        else:
            opp_hist_str = "No history yet"
            opp_tendency = "(No data)"

        # Format my history
        if context.my_history:
            my_moves = [choice.value for choice in context.my_history[-10:]]
            my_hist_str = " -> ".join(my_moves)
        else:
            my_hist_str = "No history yet"

        # Recent pattern analysis
        pattern_analysis = ""
        if len(context.opponent_history) >= 3:
            recent_3 = context.opponent_history[-3:]
            if all(choice == recent_3[0] for choice in recent_3):
                pattern_analysis = f"⚠️ Opponent stuck on {recent_3[0].value} (3 rounds)"
            elif len(set(recent_3)) == 1:
                pattern_analysis = "🔄 Opponent alternating pattern detected"

        return (
            f"=== MATCH STATE ===\n"
            f"Round: {context.round_number}\n"
            f"Score: {context.score_summary}\n"
            f"Opponent: {context.opponent_id}\n\n"

            f"=== OPPONENT ANALYSIS ===\n"
            f"History: {opp_hist_str}\n"
            f"Tendency: {opp_tendency}\n"
            f"{pattern_analysis}\n\n"

            f"=== MY HISTORY ===\n"
            f"My moves: {my_hist_str}\n\n"

            f"=== YOUR TASK ===\n"
            f"Analyze the opponent's pattern and choose your optimal move for round {context.round_number}.\n"
            f"Consider:\n"
            f"1. What pattern is the opponent following?\n"
            f"2. How can you exploit their tendencies?\n"
            f"3. Should you break your own pattern?\n"
            f"4. What's the probability-weighted best choice?\n\n"

            f"Make your strategic decision now:"
        )


class ResponseParser:
    """
    Advanced JSON parsing with multiple fallback strategies.
    Handles common LLM response formatting issues.
    """

    @staticmethod
    def extract_and_validate_json(raw_text: str) -> Dict:
        """
        Extract and validate JSON from LLM response with multiple strategies.

        Args:
            raw_text: Raw text response from LLM

        Returns:
            Parsed and validated JSON dictionary

        Raises:
            LLMError: If no valid JSON can be extracted
        """

        # Strategy 1: Direct JSON parsing (for well-formatted responses)
        try:
            data = json.loads(raw_text.strip())
            return ResponseParser._validate_response_structure(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Find JSON block in markdown or mixed content
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # Markdown JSON blocks
            r'```\s*(\{.*?\})\s*```',      # Generic code blocks
            r'\{[^{}]*"choice"[^{}]*\}',   # Simple JSON matching choice field
            r'\{.*?\}',                     # Any JSON-like structure
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    return ResponseParser._validate_response_structure(data)
                except (json.JSONDecodeError, ValueError):
                    continue

        # Strategy 3: Intelligent field extraction with regex
        choice_match = re.search(r'"choice":\s*"(even|odd)"', raw_text, re.IGNORECASE)
        confidence_match = re.search(r'"confidence":\s*([\d.]+)', raw_text)
        reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', raw_text)

        if choice_match:
            extracted_data = {
                "choice": choice_match.group(1).lower(),
                "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
                "reasoning": reasoning_match.group(1) if reasoning_match else "Extracted from unstructured response"
            }
            return ResponseParser._validate_response_structure(extracted_data)

        # Strategy 4: Simple keyword detection
        text_lower = raw_text.lower()
        if "even" in text_lower and "odd" not in text_lower:
            return {"choice": "even", "confidence": 0.4, "reasoning": "Keyword detection fallback"}
        elif "odd" in text_lower and "even" not in text_lower:
            return {"choice": "odd", "confidence": 0.4, "reasoning": "Keyword detection fallback"}

        # All strategies failed
        raise LLMError(
            f"Could not extract valid response from LLM output",
            response_data=raw_text[:200]  # First 200 chars for debugging
        )

    @staticmethod
    def _validate_response_structure(data: Dict) -> Dict:
        """
        Validate and normalize the extracted JSON structure.

        Args:
            data: Extracted JSON data

        Returns:
            Validated and normalized dictionary

        Raises:
            LLMError: If structure is invalid
        """
        if not isinstance(data, dict):
            raise LLMError("Response must be a JSON object")

        # Validate choice field
        choice = data.get("choice", "").lower().strip()
        if choice not in ["even", "odd"]:
            raise LLMError(f"Invalid choice '{choice}'. Must be 'even' or 'odd'")

        # Validate confidence field
        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
        except (ValueError, TypeError):
            confidence = 0.5

        # Validate reasoning field
        reasoning = data.get("reasoning", "No reasoning provided")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        reasoning = reasoning[:500]  # Limit length

        return {
            "choice": choice,
            "confidence": confidence,
            "reasoning": reasoning
        }


# --- Main Ollama Client ---

class OllamaClient:
    """
    Async client for Ollama API with comprehensive error handling.
    Implements timeout management, response parsing, and fallback strategies.
    """

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self.url = f"{base_url}/api/generate"

        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.avg_response_time = 0.0

    async def get_strategic_move(self, context: GameContext) -> tuple[ParityChoice, float, str]:
        """
        Get strategic move recommendation from LLM.

        Args:
            context: Game context for decision making

        Returns:
            Tuple of (choice, confidence, reasoning)

        Raises:
            LLMError: If LLM fails and no fallback can be applied
        """
        start_time = time.time()
        self.total_requests += 1

        try:
            logger.info(f"Querying Ollama model '{self.model}' for strategic decision")

            # Build prompts
            system_prompt = PromptBuilder.build_system_prompt()
            user_prompt = PromptBuilder.build_user_prompt(context)

            # Combine prompts (Ollama format)
            full_prompt = f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_prompt}\n\n<|assistant|>\n"

            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS,
                    "stop": ["<|user|>", "<|system|>"],  # Stop tokens
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }

            # Make HTTP request with timeout
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    json=payload,
                    timeout=LLM_TIMEOUT,
                    headers={"Content-Type": "application/json"}
                )

                response.raise_for_status()

            # Parse response
            response_data = response.json()
            raw_response = response_data.get("response", "")

            if not raw_response.strip():
                raise LLMError("Empty response from Ollama", model_name=self.model)

            logger.debug(f"Raw LLM response: {raw_response[:100]}...")

            # Extract and validate JSON
            parsed_data = ResponseParser.extract_and_validate_json(raw_response)

            # Convert to expected types
            choice = ParityChoice(parsed_data["choice"])
            confidence = parsed_data["confidence"]
            reasoning = parsed_data["reasoning"]

            # Update performance metrics
            response_time = (time.time() - start_time) * 1000
            self.successful_requests += 1
            self._update_avg_response_time(response_time)

            log_performance(
                logger=logger,
                operation="llm_decision",
                duration_ms=response_time,
                success=True,
                model=self.model,
                choice=choice.value,
                confidence=confidence
            )

            logger.info(f"LLM Decision: {choice.value} (confidence: {confidence:.2f}) - {reasoning}")
            return choice, confidence, reasoning

        except httpx.TimeoutException:
            error_msg = f"Ollama request timed out after {LLM_TIMEOUT}s"
            logger.error(error_msg)
            return self._fallback_strategy("timeout", context)

        except httpx.ConnectError:
            error_msg = "Cannot connect to Ollama server"
            logger.error(error_msg)
            return self._fallback_strategy("connection_error", context)

        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama HTTP error: {e.response.status_code}"
            logger.error(error_msg)
            return self._fallback_strategy("http_error", context)

        except LLMError as e:
            logger.error(f"LLM parsing error: {e}")
            return self._fallback_strategy("parsing_error", context)

        except Exception as e:
            logger.error(f"Unexpected LLM error: {str(e)}", exc_info=True)
            return self._fallback_strategy("unknown_error", context)

    def _fallback_strategy(self, error_type: str, context: GameContext) -> tuple[ParityChoice, float, str]:
        """
        Deterministic fallback strategy when LLM fails.
        Uses game theory principles for reasonable choices.

        Args:
            error_type: Type of error that triggered fallback
            context: Game context for intelligent fallback

        Returns:
            Tuple of (choice, confidence, reasoning)
        """

        # Strategy 1: Counter-opponent if we have enough history
        if len(context.opponent_history) >= 3:
            recent_moves = context.opponent_history[-3:]
            even_count = sum(1 for choice in recent_moves if choice == ParityChoice.EVEN)

            if even_count >= 2:  # Opponent favors EVEN
                choice = ParityChoice.ODD
                confidence = 0.6
                reasoning = f"Fallback: Counter opponent's EVEN tendency ({error_type})"
            else:  # Opponent favors ODD
                choice = ParityChoice.EVEN
                confidence = 0.6
                reasoning = f"Fallback: Counter opponent's ODD tendency ({error_type})"

        # Strategy 2: Nash equilibrium (random) if no clear pattern
        else:
            import random
            choice = random.choice([ParityChoice.EVEN, ParityChoice.ODD])
            confidence = 0.5
            reasoning = f"Fallback: Nash equilibrium strategy ({error_type})"

        logger.warning(f"Using fallback strategy: {choice.value} due to {error_type}")
        return choice, confidence, reasoning

    def _update_avg_response_time(self, response_time_ms: float):
        """Update rolling average response time"""
        if self.successful_requests == 1:
            self.avg_response_time = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_response_time = (alpha * response_time_ms) + ((1 - alpha) * self.avg_response_time)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        success_rate = (self.successful_requests / self.total_requests) if self.total_requests > 0 else 0.0

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round(success_rate, 3),
            "avg_response_time_ms": round(self.avg_response_time, 2),
            "model": self.model
        }


# --- Integration Testing ---

async def test_ollama_integration():
    """Test function for Ollama integration"""
    try:
        client = OllamaClient()

        # Create test context
        test_context = GameContext(
            agent_id="player:TEST",
            opponent_id="player:OPPONENT",
            match_id="M-TEST",
            round_number=3,
            opponent_history=[ParityChoice.EVEN, ParityChoice.ODD, ParityChoice.EVEN],
            my_history=[ParityChoice.ODD, ParityChoice.EVEN, ParityChoice.ODD],
            current_score={"player:TEST": 1, "player:OPPONENT": 2}
        )

        # Test LLM decision
        choice, confidence, reasoning = await client.get_strategic_move(test_context)

        print(f"✅ LLM Test Result: {choice.value} (confidence: {confidence:.2f})")
        print(f"   Reasoning: {reasoning}")
        print(f"   Performance: {client.get_performance_stats()}")

        return True

    except Exception as e:
        print(f"❌ LLM Test Failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Simple CLI test
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        result = asyncio.run(test_ollama_integration())
        sys.exit(0 if result else 1)
    else:
        print("Ollama Strategy Module - Use 'python ollama_strategy.py test' to run tests")