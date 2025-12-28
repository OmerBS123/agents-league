import httpx
import asyncio
import logging
import json
import re
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
TIMEOUT_SECONDS = 3.0 # Strict timeout for game loops
MAX_VALIDATION_ATTEMPTS = 2

# --- Logging ---
logger = logging.getLogger("OllamaStrategy")
logging.basicConfig(level=logging.INFO)

# --- Data Structures ---

@dataclass
class GameContext:
    """Holds the history required to build a prompt"""
    opponent_id: str
    round_number: int
    opponent_history: List[str] # ["even", "odd", "even"]
    my_history: List[str]
    current_score: str # "3-2"

class AIResponseError(Exception):
    """Raised when the LLM output is unusable"""
    pass

# --- Helper Classes ---

class PromptBuilder:
    """
    Constructs optimized prompts for the specific model.
    Separating this allows easy swapping of prompt engineering techniques.
    """
    @staticmethod
    def build_system_prompt() -> str:
        return (
            "You are a strategic Game Theory AI playing the 'Even/Odd' game. "
            "Your Goal: Win the match by predicting the winning parity. "
            "Rules: "
            "1. Two players choose 'even' or 'odd'. "
            "2. A random number (1-10) is drawn. "
            "3. If (Random + Your Choice) % 2 == 0, you might win (simplified). "
            "Output Requirement: Respond with ONLY a JSON object: {\"choice\": \"even\", \"reason\": \"...\"}."
        )

    @staticmethod
    def build_user_prompt(context: GameContext) -> str:
        # Format history for the LLM to analyze patterns
        opp_hist_str = ", ".join(context.opponent_history[-5:]) if context.opponent_history else "None"
        
        return (
            f"--- Game State ---\n"
            f"Opponent: {context.opponent_id}\n"
            f"Round: {context.round_number}\n"
            f"Opponent Past Moves (Last 5): [{opp_hist_str}]\n"
            f"------------------\n"
            f"Analyze the opponent's pattern. Do they alternate? Do they stick to one choice?\n"
            f"Based on this, make your move."
        )

class ResponseParser:
    """
    Cleans and validates the raw text from the LLM.
    LLMs often add preamble text ("Here is your JSON...") which breaks parsing.
    """
    @staticmethod
    def extract_json(raw_text: str) -> Dict:
        """Finds the first valid JSON object in a string"""
        # 1. Try direct parsing
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 2. Regex search for { ... }
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise AIResponseError(f"Could not extract JSON from: {raw_text[:50]}...")

    @staticmethod
    def validate_move(data: Dict) -> Literal["even", "odd"]:
        """Ensures the JSON contains the required fields"""
        choice = data.get("choice", "").lower().strip()
        if choice not in ["even", "odd"]:
            raise AIResponseError(f"Invalid choice found: '{choice}'")
        return choice

# --- Main Client ---

class OllamaClient:
    """
    Manages the connection to the Ollama API.
    Handles timeouts, parsing, and fallbacks.
    """
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.url = f"{OLLAMA_BASE_URL}/api/generate"

    async def get_move(self, context: GameContext) -> str:
        """
        The main public method. 
        Guarantees a return value ("even" or "odd") even if AI fails.
        """
        system_prompt = PromptBuilder.build_system_prompt()
        user_prompt = PromptBuilder.build_user_prompt(context)
        
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7, # Some creativity for randomness
                "num_predict": 100  # Keep response short
            }
        }

        logger.info(f"Querying Ollama ({self.model})...")
        
        try:
            # 1. Network Call
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.url, 
                    json=payload, 
                    timeout=TIMEOUT_SECONDS
                )
                resp.raise_for_status()
                
            # 2. Parsing
            raw_response = resp.json().get("response", "")
            logger.debug(f"Raw AI Response: {raw_response}")
            
            json_data = ResponseParser.extract_json(raw_response)
            move = ResponseParser.validate_move(json_data)
            reason = json_data.get("reason", "No reason provided")
            
            logger.info(f"AI Decision: {move.upper()} (Reason: {reason})")
            return move

        except (httpx.TimeoutException, httpx.ConnectError):
            logger.error("Ollama connection timed out/failed.")
            return self._fallback_strategy("connection_error")
            
        except AIResponseError as e:
            logger.error(f"AI Response Validation Failed: {e}")
            return self._fallback_strategy("validation_error")
            
        except Exception as e:
            logger.error(f"Unexpected error in LLM strategy: {e}")
            return self._fallback_strategy("unknown_error")

    def _fallback_strategy(self, reason: str) -> str:
        """
        Deterministic fallback to ensure the game continues.
        Always defaults to 'even' as a safe bet, or random.
        """
        import random
        choice = random.choice(["even", "odd"])
        logger.warning(f"Using FALLBACK strategy ({reason}) -> {choice}")
        return choice

# --- Usage Simulation ---

async def main():
    # 1. Setup Mock Context
    ctx = GameContext(
        opponent_id="player:P05",
        round_number=3,
        opponent_history=["even", "even", "odd", "even"],
        my_history=["odd", "odd", "odd", "even"],
        current_score="1-1"
    )

    # 2. Init Client
    client = OllamaClient(model="llama3")

    # 3. Execute
    print("🤖 Analyzing Game State...")
    move = await client.get_move(ctx)
    print(f"🏁 Final Move: {move}")

if __name__ == "__main__":
    # Check if Ollama is actually running before waiting
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")