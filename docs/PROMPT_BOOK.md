# Prompt Book

## Distributed AI Agent League System

This document catalogs all prompts and AI interactions used in the development and operation of the Distributed AI Agent League System.

---

## 1. LLM Strategy Prompts

### 1.1 System Prompt (Game Theory AI)

**Location:** `shared/ollama_strategy.py` - `PromptBuilder.build_system_prompt()`

**Purpose:** Define the LLM's role and constraints for game decision-making

```
You are an expert Game Theory AI playing the 'Even/Odd' number game.
Your goal is to maximize your wins by making optimal strategic decisions.

GAME RULES:
1. Two players simultaneously choose 'even' or 'odd'
2. A random number (1-10) is generated
3. The sum of your choice value + opponent choice value + random number determines the winner
4. Choice values: 'odd' = 1, 'even' = 0
5. If the final sum is even, players who chose 'even' win that round
6. If the final sum is odd, players who chose 'odd' win that round
7. First to win majority of rounds wins the match

STRATEGIC CONSIDERATIONS:
- Analyze opponent patterns and tendencies
- Consider probability distributions of random numbers
- Use game theory concepts like Nash equilibrium
- Adapt your strategy based on match progress
- Consider psychological factors and pattern breaking

OUTPUT REQUIREMENT:
Respond with ONLY a valid JSON object in this exact format:
{
  "choice": "even",
  "confidence": 0.75,
  "reasoning": "Brief explanation of your strategic reasoning"
}

CONSTRAINTS:
- Choice must be exactly 'even' or 'odd'
- Confidence must be a number between 0.0 and 1.0
- Reasoning must be concise (under 100 characters)
- No additional text outside the JSON object
```

**Design Rationale:**
- Clear role definition ("expert Game Theory AI")
- Explicit game rules to prevent hallucination
- Structured output format for reliable parsing
- Confidence score for decision quality tracking
- Length constraints for fast inference

---

### 1.2 User Prompt Template (Per-Move Context)

**Location:** `shared/ollama_strategy.py` - `PromptBuilder.build_user_prompt()`

**Purpose:** Provide game state context for each move decision

```
=== MATCH STATE ===
Round: {round_number}
Score: {score_summary}
Opponent: {opponent_id}

=== OPPONENT ANALYSIS ===
History: {opponent_history}  # e.g., "even -> odd -> even -> odd -> even"
Tendency: {tendency}         # e.g., "(3 even, 2 odd)"
{pattern_warning}            # e.g., "⚠️ Opponent stuck on even (3 rounds)"

=== MY HISTORY ===
My moves: {my_history}

=== YOUR TASK ===
Analyze the opponent's pattern and choose your optimal move for round {round_number}.
Consider:
1. What pattern is the opponent following?
2. How can you exploit their tendencies?
3. Should you break your own pattern?
4. What's the probability-weighted best choice?

Make your strategic decision now:
```

**Design Rationale:**
- Structured sections for easy parsing by LLM
- Historical context for pattern recognition
- Explicit analysis prompts for chain-of-thought
- Emoji indicators for pattern warnings

---

### 1.3 Example LLM Responses

**Successful Response:**
```json
{
  "choice": "odd",
  "confidence": 0.82,
  "reasoning": "Opponent shows alternating pattern, predicting even next, countering with odd"
}
```

**Fallback-Triggering Response:**
```
Based on my analysis of the game state, I would recommend...

{
  "choice": "even",
  "confidence": 0.6,
  "reasoning": "Counter opponent tendency"
}

This is because the opponent has shown a preference for odd choices.
```

*Note: The system extracts JSON from mixed content using regex patterns*

---

## 2. Development Prompts

### 2.1 Initial Architecture Design

**Prompt used with Claude:**
```
I need to design a distributed multi-agent system for a simple game tournament.
Requirements:
- Multiple autonomous agents communicating over HTTP
- Central coordinator for tournament management
- Referee for match orchestration
- Pluggable player strategies (random, pattern-based, LLM)
- Pydantic for message validation
- FastAPI for agent APIs

Please design the architecture with:
1. Component responsibilities
2. Communication protocol
3. Data models
4. Error handling strategy
```

**Outcome:** Initial system architecture with League Manager, Referee, and Player components

---

### 2.2 Schema Design Iteration

**Prompt used:**
```
I have an MCP envelope schema that needs cross-field validation.
The message_type field should determine what structure is valid in the data field.
For example:
- message_type: "REGISTER" -> data must have display_name, contact_endpoint
- message_type: "MOVE_RESPONSE" -> data must have parity_choice, confidence

How do I implement this with Pydantic v2?
```

**Outcome:** Model validator pattern in `MCPEnvelope.validate_data_content()`

---

### 2.3 Strategy Pattern Implementation

**Prompt used:**
```
Implement the Strategy pattern for player decision-making in Python with:
1. Abstract base class with choose_move() method
2. RandomStrategy - uniform random
3. HistoryStrategy - analyze opponent patterns
4. LLMStrategy - integrate with Ollama

Requirements:
- Async methods for LLM calls
- Return tuple of (choice, confidence, reasoning)
- Fallback to random if LLM fails
```

**Outcome:** Strategy hierarchy in `agents/player.py`

---

### 2.4 Error Handling Design

**Prompt used:**
```
Design an exception hierarchy for a distributed agent system with:
- Base LeagueError
- Specific errors for: registration, match execution, strategy, network, timeout, LLM, protocol
- Each error should carry relevant context
- Errors should be recoverable vs non-recoverable
```

**Outcome:** Exception hierarchy in `shared/exceptions.py`

---

### 2.5 Logging Strategy

**Prompt used:**
```
Design a logging system for a multi-agent distributed system that:
1. Uses structured JSON format
2. Includes request IDs for tracing
3. Has contextual logging with agent_id, match_id
4. Includes performance metrics logging
5. Works with Python's logging module
```

**Outcome:** Logging utilities in `shared/logger.py`

---

## 3. Prompt Engineering Techniques Used

### 3.1 JSON-Only Output

**Technique:** Strict output format specification

```
OUTPUT REQUIREMENT:
Respond with ONLY a valid JSON object...
No additional text outside the JSON object
```

**Why:** Ensures parseable responses, reduces post-processing complexity

---

### 3.2 Chain-of-Thought Prompting

**Technique:** Explicit analysis steps before decision

```
Consider:
1. What pattern is the opponent following?
2. How can you exploit their tendencies?
3. Should you break your own pattern?
4. What's the probability-weighted best choice?
```

**Why:** Improves decision quality by forcing structured reasoning

---

### 3.3 Role Assignment

**Technique:** Clear persona definition

```
You are an expert Game Theory AI...
```

**Why:** Activates relevant knowledge patterns in the LLM

---

### 3.4 Constraint Specification

**Technique:** Explicit output constraints

```
CONSTRAINTS:
- Choice must be exactly 'even' or 'odd'
- Confidence must be a number between 0.0 and 1.0
- Reasoning must be concise (under 100 characters)
```

**Why:** Prevents invalid outputs, enables validation

---

### 3.5 Context Injection

**Technique:** Dynamic game state in prompts

```
Round: {round_number}
Score: {score_summary}
History: {opponent_history}
```

**Why:** Provides relevant context for informed decisions

---

## 4. Prompt Versioning

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-12 | Initial prompt design |
| v1.1 | 2025-12 | Added pattern warning indicators |
| v1.2 | 2026-01 | Improved JSON extraction robustness |

---

## 5. Known Prompt Issues and Mitigations

### 5.1 Verbose Responses

**Issue:** LLM sometimes adds explanation outside JSON

**Mitigation:** Multi-strategy JSON extraction in `ResponseParser.extract_and_validate_json()`

### 5.2 Confidence Hallucination

**Issue:** LLM reports high confidence for random guesses

**Mitigation:** Treat confidence as informational, don't use for decision weighting

### 5.3 Slow Inference

**Issue:** LLM responses can take 1-3 seconds

**Mitigation:**
- 3-second timeout with fallback
- Fallback uses history-based strategy
- Performance metrics logging for monitoring

---

## 6. Future Prompt Improvements

1. **Few-Shot Examples** - Add example game sequences with optimal play
2. **Self-Consistency** - Multiple inference passes with voting
3. **Reflection** - Ask LLM to critique its own reasoning
4. **Memory** - Include more historical context across matches
5. **Model Tuning** - Fine-tune on game theory datasets
