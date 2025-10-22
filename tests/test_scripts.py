"""
Tests for start.sh and stop.sh scripts.
"""
import os
import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def start_script(project_root):
    """Get start.sh script path."""
    return project_root / "start.sh"


@pytest.fixture
def stop_script(project_root):
    """Get stop.sh script path."""
    return project_root / "stop.sh"


@pytest.fixture
def pid_file(project_root):
    """Get PID file path."""
    return project_root / "streamlit.pid"


class TestScriptExistence:
    """Test that scripts exist and are executable."""
    
    def test_start_script_exists(self, start_script):
        """Test that start.sh exists."""
        assert start_script.exists(), f"start.sh not found at {start_script}"
    
    def test_stop_script_exists(self, stop_script):
        """Test that stop.sh exists."""
        assert stop_script.exists(), f"stop.sh not found at {stop_script}"
    
    def test_start_script_executable(self, start_script):
        """Test that start.sh is executable."""
        assert os.access(start_script, os.X_OK), "start.sh is not executable"
    
    def test_stop_script_executable(self, stop_script):
        """Test that stop.sh is executable."""
        assert os.access(stop_script, os.X_OK), "stop.sh is not executable"


class TestScriptContent:
    """Test script content and structure."""
    
    def test_start_script_has_shebang(self, start_script):
        """Test that start.sh has proper shebang."""
        with open(start_script, 'r') as f:
            first_line = f.readline().strip()
        assert first_line == "#!/bin/bash", "start.sh missing proper shebang"
    
    def test_stop_script_has_shebang(self, stop_script):
        """Test that stop.sh has proper shebang."""
        with open(stop_script, 'r') as f:
            first_line = f.readline().strip()
        assert first_line == "#!/bin/bash", "stop.sh missing proper shebang"
    
    def test_start_script_references_pid_file(self, start_script):
        """Test that start.sh references PID file."""
        with open(start_script, 'r') as f:
            content = f.read()
        assert "streamlit.pid" in content, "start.sh doesn't reference PID file"
        assert "PID_FILE" in content, "start.sh doesn't define PID_FILE variable"
    
    def test_stop_script_references_pid_file(self, stop_script):
        """Test that stop.sh references PID file."""
        with open(stop_script, 'r') as f:
            content = f.read()
        assert "streamlit.pid" in content, "stop.sh doesn't reference PID file"
        assert "PID_FILE" in content, "stop.sh doesn't define PID_FILE variable"


class TestPIDFileHandling:
    """Test PID file creation and cleanup."""
    
    def test_stop_script_handles_missing_pid_file(self, stop_script, pid_file, project_root):
        """Test that stop.sh handles missing PID file gracefully."""
        # Ensure PID file doesn't exist
        if pid_file.exists():
            pid_file.unlink()
        
        # Run stop script
        result = subprocess.run(
            [str(stop_script)],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        # Should exit with error code but not crash
        assert result.returncode != 0, "stop.sh should fail when no PID file exists"
        assert "No PID file found" in result.stdout or "not running" in result.stdout.lower()
    
    def test_stop_script_handles_invalid_pid(self, stop_script, pid_file, project_root):
        """Test that stop.sh handles invalid PID gracefully."""
        # Create PID file with invalid content
        with open(pid_file, 'w') as f:
            f.write("invalid_pid")
        
        try:
            # Run stop script
            result = subprocess.run(
                [str(stop_script)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Should handle invalid PID
            assert result.returncode != 0, "stop.sh should fail with invalid PID"
            assert "Invalid PID" in result.stdout or "not running" in result.stdout.lower()
            
            # PID file should be cleaned up
            assert not pid_file.exists(), "PID file should be removed after invalid PID"
        finally:
            # Cleanup
            if pid_file.exists():
                pid_file.unlink()
    
    def test_stop_script_handles_stale_pid(self, stop_script, pid_file, project_root):
        """Test that stop.sh handles stale PID (process not running)."""
        # Create PID file with non-existent PID
        with open(pid_file, 'w') as f:
            f.write("999999")
        
        try:
            # Run stop script
            result = subprocess.run(
                [str(stop_script)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Should handle stale PID
            assert result.returncode != 0, "stop.sh should fail with stale PID"
            assert "not running" in result.stdout.lower() or "stale" in result.stdout.lower()
            
            # PID file should be cleaned up
            assert not pid_file.exists(), "PID file should be removed after stale PID"
        finally:
            # Cleanup
            if pid_file.exists():
                pid_file.unlink()


class TestScriptIntegration:
    """Integration tests for script behavior."""
    
    def test_start_script_creates_log_directory(self, start_script, project_root):
        """Test that start.sh creates logs directory."""
        log_dir = project_root / "logs"
        
        # Note: We don't actually start Streamlit in tests
        # Just verify the script would create the directory
        with open(start_script, 'r') as f:
            content = f.read()
        
        assert "mkdir -p" in content, "start.sh should create logs directory"
        assert "LOG_DIR" in content, "start.sh should define LOG_DIR variable"
    
    def test_start_script_validates_app_path(self, start_script):
        """Test that start.sh validates app.py exists."""
        with open(start_script, 'r') as f:
            content = f.read()
        
        assert "APP_PATH" in content, "start.sh should define APP_PATH"
        assert "-f" in content and "APP_PATH" in content, "start.sh should check if app file exists"
    
    def test_scripts_use_project_root(self, start_script, stop_script):
        """Test that scripts properly determine project root."""
        for script in [start_script, stop_script]:
            with open(script, 'r') as f:
                content = f.read()
            
            assert "PROJECT_ROOT" in content, f"{script.name} should define PROJECT_ROOT"
            assert "dirname" in content, f"{script.name} should use dirname to find project root"
