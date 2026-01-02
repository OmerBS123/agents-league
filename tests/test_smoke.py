"""
Smoke tests - Basic sanity checks that should always pass.
These tests verify the project structure and basic imports work.
"""

import os
import pytest


class TestProjectStructure:
    """Tests that verify the project structure is correct."""

    def test_agents_folder_exists(self):
        """Test that the agents folder exists."""
        assert os.path.isdir("agents"), "agents/ directory should exist"

    def test_shared_folder_exists(self):
        """Test that the shared folder exists."""
        assert os.path.isdir("shared"), "shared/ directory should exist"

    def test_consts_folder_exists(self):
        """Test that the consts folder exists."""
        assert os.path.isdir("consts"), "consts/ directory should exist"

    def test_tests_folder_exists(self):
        """Test that the tests folder exists."""
        assert os.path.isdir("tests"), "tests/ directory should exist"

    def test_docs_folder_exists(self):
        """Test that the docs folder exists."""
        assert os.path.isdir("docs"), "docs/ directory should exist"


class TestRequiredFiles:
    """Tests that verify required files exist."""

    def test_readme_exists(self):
        """Test that README.md exists."""
        assert os.path.isfile("README.md"), "README.md should exist"

    def test_pyproject_exists(self):
        """Test that pyproject.toml exists."""
        assert os.path.isfile("pyproject.toml"), "pyproject.toml should exist"

    def test_gitignore_exists(self):
        """Test that .gitignore exists."""
        assert os.path.isfile(".gitignore"), ".gitignore should exist"

    def test_agents_init_exists(self):
        """Test that agents/__init__.py exists."""
        assert os.path.isfile("agents/__init__.py"), "agents/__init__.py should exist"

    def test_shared_init_exists(self):
        """Test that shared/__init__.py exists."""
        assert os.path.isfile("shared/__init__.py"), "shared/__init__.py should exist"


class TestBasicImports:
    """Tests that verify basic imports work."""

    def test_import_schemas(self):
        """Test that schemas module can be imported."""
        from shared import schemas
        assert schemas is not None

    def test_import_exceptions(self):
        """Test that exceptions module can be imported."""
        from shared import exceptions
        assert exceptions is not None

    def test_import_logger(self):
        """Test that logger module can be imported."""
        from shared import logger
        assert logger is not None

    def test_import_consts(self):
        """Test that consts module can be imported."""
        import consts
        assert consts is not None

    def test_import_parity_choice(self):
        """Test that ParityChoice enum can be imported."""
        from shared.schemas import ParityChoice
        assert ParityChoice.EVEN.value == "even"
        assert ParityChoice.ODD.value == "odd"

    def test_import_message_type(self):
        """Test that MessageType enum can be imported."""
        from shared.schemas import MessageType
        assert MessageType.REGISTER is not None

    def test_import_league_error(self):
        """Test that LeagueError can be imported."""
        from shared.exceptions import LeagueError
        assert LeagueError is not None
