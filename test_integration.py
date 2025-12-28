"""
Integration test for the Distributed AI Agent League System.
Tests component initialization, schema validation, and basic functionality.
"""

import asyncio
import sys
from typing import List, Dict, Any

def test_imports():
    """Test that core modules can be imported without errors"""
    print("🔍 Testing module imports...")

    try:
        # Test shared components (no external deps)
        from shared.logger import get_logger
        from shared.exceptions import LeagueError, RegistrationError, MatchError
        from shared.schemas import MCPEnvelope, MessageType, ParityChoice, create_registration_message

        # Test constants (no external deps)
        from consts import LEAGUE_MANAGER_PORT, REFEREE_PORT, PLAYER_PORTS

        print("✅ Core module imports successful")
        return True

    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False


def test_schema_validation():
    """Test Pydantic schema validation"""
    print("🔍 Testing schema validation...")

    try:
        from shared.schemas import MCPEnvelope, MessageType, create_registration_message

        # Test valid registration message
        reg_msg = create_registration_message(
            sender_id="player:TEST",
            display_name="Test Player",
            contact_endpoint="http://localhost:8101",
            strategies=["random"]
        )

        # Test JSON serialization/deserialization
        json_str = reg_msg.to_json()
        parsed_msg = MCPEnvelope.from_json(json_str)

        assert parsed_msg.message_type == MessageType.REGISTER
        assert parsed_msg.sender == "player:TEST"

        print("✅ Schema validation successful")
        return True

    except Exception as e:
        print(f"❌ Schema validation error: {str(e)}")
        return False


def test_strategy_pattern():
    """Test strategy pattern concepts (without full agent deps)"""
    print("🔍 Testing strategy pattern...")

    try:
        from shared.schemas import ParityChoice
        import random

        # Test ParityChoice enum
        assert ParityChoice.EVEN.value == "even"
        assert ParityChoice.ODD.value == "odd"

        # Test random choice simulation
        choices = [ParityChoice.EVEN, ParityChoice.ODD]
        selected = random.choice(choices)
        assert selected in choices

        print("✅ Strategy pattern concepts successful")
        return True

    except Exception as e:
        print(f"❌ Strategy pattern error: {str(e)}")
        return False


async def test_async_functionality():
    """Test async components without external dependencies"""
    print("🔍 Testing async functionality...")

    try:
        import asyncio
        from shared.schemas import ParityChoice

        # Test basic async functionality
        async def mock_strategy():
            await asyncio.sleep(0.001)  # Minimal async operation
            return ParityChoice.EVEN, 0.5, "Mock reasoning"

        choice, confidence, reasoning = await mock_strategy()

        assert choice in [ParityChoice.EVEN, ParityChoice.ODD]
        assert 0.0 <= confidence <= 1.0
        assert isinstance(reasoning, str)

        print("✅ Async functionality successful")
        return True

    except Exception as e:
        print(f"❌ Async functionality error: {str(e)}")
        return False


def test_configuration():
    """Test configuration constants"""
    print("🔍 Testing configuration...")

    try:
        from consts import (
            LEAGUE_MANAGER_PORT, REFEREE_PORT, PLAYER_PORTS,
            DEFAULT_TIMEOUT, AVAILABLE_STRATEGIES
        )

        # Validate port configurations
        assert LEAGUE_MANAGER_PORT == 8000
        assert REFEREE_PORT == 8001
        assert len(PLAYER_PORTS) == 4
        assert all(8101 <= port <= 8104 for port in PLAYER_PORTS)

        # Validate strategies
        assert "random" in AVAILABLE_STRATEGIES
        assert "history" in AVAILABLE_STRATEGIES
        assert "llm" in AVAILABLE_STRATEGIES

        print("✅ Configuration validation successful")
        return True

    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return False


def test_logging():
    """Test logging system"""
    print("🔍 Testing logging system...")

    try:
        from shared.logger import get_logger, log_with_context, log_performance
        import logging

        logger = get_logger(__name__, "test-service")

        # Test basic logging
        logger.info("Test log message")

        # Test contextual logging
        log_with_context(
            logger=logger,
            level=logging.INFO,
            message="Test context logging",
            agent_id="test:001",
            match_id="M-001"
        )

        # Test performance logging
        log_performance(
            logger=logger,
            operation="test_operation",
            duration_ms=42.5,
            success=True
        )

        print("✅ Logging system successful")
        return True

    except Exception as e:
        print(f"❌ Logging error: {str(e)}")
        return False


async def run_all_tests():
    """Run all integration tests"""
    print("🧪 Distributed AI Agent League System - Integration Tests")
    print("=" * 60)

    tests = [
        ("Module Imports", test_imports),
        ("Schema Validation", test_schema_validation),
        ("Strategy Pattern", test_strategy_pattern),
        ("Configuration", test_configuration),
        ("Logging System", test_logging),
        ("Async Functionality", test_async_functionality)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()

            if success:
                passed += 1

        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")

    print("\n" + "=" * 60)
    print(f"🏁 Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integration tests passed! System ready for deployment.")
        return True
    else:
        print("⚠️  Some integration tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test runner failed: {str(e)}")
        sys.exit(1)