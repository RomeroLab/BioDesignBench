"""Docker-based code execution sandbox."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ExecutionResult(BaseModel):
    """Result from code execution."""

    success: bool
    return_code: int
    stdout: str
    stderr: str
    output_files: list[str] = []
    execution_time_seconds: float = 0.0


class DockerExecutor:
    """Execute code in a Docker container for isolation.

    Falls back to subprocess execution if Docker is unavailable.
    """

    def __init__(
        self,
        image: str = "biodesignbench-sandbox",
        timeout_seconds: int = 300,
        memory_limit: str = "4g",
        network_disabled: bool = True,
        project_root: Path | None = None,
    ):
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.network_disabled = network_disabled
        self.project_root = project_root or self._detect_project_root()
        self._docker_client: Any | None = None
        self._docker_available: bool | None = None

    @staticmethod
    def _detect_project_root() -> Path:
        """Detect project root by looking for pyproject.toml."""
        candidate = Path(__file__).resolve().parents[2]
        if (candidate / "pyproject.toml").exists():
            return candidate
        return Path.cwd()

    @property
    def docker_available(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is None:
            try:
                import docker

                client = docker.from_env()
                client.ping()
                self._docker_client = client
                self._docker_available = True
            except Exception:
                self._docker_available = False
        return self._docker_available

    def execute(
        self,
        code: str,
        working_dir: Path | None = None,
        input_files: dict[str, str | bytes] | None = None,
    ) -> ExecutionResult:
        """Execute Python code and return results.

        Args:
            code: Python code to execute.
            working_dir: Directory to use for I/O. If None, a temp dir is created.
            input_files: Dict of {filename: content} to place in working dir.

        Returns:
            ExecutionResult with stdout, stderr, output files, etc.
        """
        cleanup_dir = False
        if working_dir is None:
            working_dir = Path(tempfile.mkdtemp(prefix="biodesign_"))
            cleanup_dir = True
        else:
            working_dir = Path(working_dir)
            working_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Write input files
            if input_files:
                for filename, content in input_files.items():
                    filepath = working_dir / filename
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, bytes):
                        filepath.write_bytes(content)
                    else:
                        filepath.write_text(content)

            # Write the script
            script_path = working_dir / "script.py"
            script_path.write_text(code)

            # Execute
            if self.docker_available:
                return self._execute_docker(working_dir)
            else:
                logger.warning("Docker unavailable, falling back to subprocess execution")
                return self._execute_subprocess(working_dir)
        finally:
            if cleanup_dir:
                shutil.rmtree(working_dir, ignore_errors=True)

    def execute_in_dir(self, code: str, output_dir: Path) -> ExecutionResult:
        """Execute code with output_dir as persistent working directory.

        Unlike execute(), output_dir is NOT cleaned up after execution.
        Output files remain in output_dir for later evaluation.

        Args:
            code: Python code to execute.
            output_dir: Directory for code I/O (persists after execution).

        Returns:
            ExecutionResult with execution details.
        """
        return self.execute(code=code, working_dir=output_dir)

    def _build_data_volumes(self) -> dict[str, dict[str, str]]:
        """Build Docker volume mounts for input data only.

        Mounts ``data/tier{1,2}/input/`` as read-only at
        ``/workspace/data/tier{1,2}/input/``.  Deliberately excludes
        ``ground_truth/``, ``prompts/``, and other directories to
        prevent agents from reading answer files.
        """
        volumes: dict[str, dict[str, str]] = {}
        for tier in ("tier1", "tier2"):
            input_dir = self.project_root / "data" / tier / "input"
            if input_dir.exists():
                volumes[str(input_dir.resolve())] = {
                    "bind": f"/workspace/data/{tier}/input",
                    "mode": "ro",
                }
        return volumes

    def _provision_input_symlinks(self, working_dir: Path) -> None:
        """Create per-tier input symlinks in working_dir.

        Creates ``working_dir/data/tier{1,2}/input/`` symlinks pointing
        to the real input directories.  Deliberately does NOT symlink the
        entire ``data/`` tree, which would expose ``ground_truth/`` and
        ``prompts/`` to the agent.
        """
        for tier in ("tier1", "tier2"):
            input_src = self.project_root / "data" / tier / "input"
            if not input_src.exists():
                continue

            input_dst = working_dir / "data" / tier / "input"
            if input_dst.exists() or input_dst.is_symlink():
                continue

            input_dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(input_src.resolve(), input_dst)
            except OSError:
                pass

    def _execute_docker(self, working_dir: Path) -> ExecutionResult:
        """Execute code in a Docker container."""
        import docker

        client = self._docker_client
        start_time = time.time()

        # Record files before execution to detect new outputs
        pre_files = set(p.name for p in working_dir.iterdir())

        # Build volume mounts: workspace + input-only data directories (read-only)
        # Deliberately excludes ground_truth/, prompts/, and other sensitive dirs.
        volumes = {
            str(working_dir.resolve()): {"bind": "/workspace", "mode": "rw"},
        }
        volumes.update(self._build_data_volumes())

        try:
            # Run as current user so output files are owned by the caller, not root
            uid_gid = f"{os.getuid()}:{os.getgid()}"

            container = client.containers.run(
                image=self.image,
                command=["python", "script.py"],
                volumes=volumes,
                working_dir="/workspace",
                user=uid_gid,
                network_mode="none" if self.network_disabled else "bridge",
                mem_limit=self.memory_limit,
                detach=True,
                stderr=True,
            )

            try:
                result = container.wait(timeout=self.timeout_seconds)
                return_code = result.get("StatusCode", -1)
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            except Exception:
                # Timeout or other error
                container.kill()
                return ExecutionResult(
                    success=False,
                    return_code=-1,
                    stdout="",
                    stderr=f"Execution timed out after {self.timeout_seconds}s",
                    execution_time_seconds=time.time() - start_time,
                )
            finally:
                container.remove(force=True)

        except docker.errors.ImageNotFound:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Docker image '{self.image}' not found. Run: docker build -t {self.image} docker/sandbox/",
                execution_time_seconds=time.time() - start_time,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Docker execution error: {e}",
                execution_time_seconds=time.time() - start_time,
            )

        # Collect output files (new files created during execution)
        post_files = set(p.name for p in working_dir.iterdir())
        new_files = sorted(post_files - pre_files - {"script.py"})

        execution_time = time.time() - start_time
        return ExecutionResult(
            success=return_code == 0,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            output_files=new_files,
            execution_time_seconds=execution_time,
        )

    _unshare_available: bool | None = None

    @classmethod
    def _check_unshare_available(cls) -> bool:
        """Test whether ``unshare --net`` is usable on this system.

        Cached after the first call to avoid repeated subprocess spawns.
        """
        if cls._unshare_available is not None:
            return cls._unshare_available

        import platform

        if platform.system() != "Linux":
            cls._unshare_available = False
            return False

        if not shutil.which("unshare"):
            cls._unshare_available = False
            return False

        # Probe: run a trivial command under unshare to see if the kernel allows it
        try:
            probe = subprocess.run(
                ["unshare", "--net", "--map-root-user", "--", "true"],
                capture_output=True,
                timeout=5,
            )
            cls._unshare_available = probe.returncode == 0
        except Exception:
            cls._unshare_available = False

        return cls._unshare_available

    @classmethod
    def _wrap_with_network_isolation(cls, cmd: list[str]) -> list[str]:
        """Wrap a command with Linux network namespace isolation.

        Uses ``unshare --net`` to create a new network namespace where
        only loopback is available.  Falls back gracefully when unshare
        is unavailable (non-Linux, missing binary, or insufficient
        kernel permissions).
        """
        if cls._check_unshare_available():
            return ["unshare", "--net", "--map-root-user", "--"] + cmd

        logger.warning(
            "Network isolation via unshare is unavailable on this system. "
            "Subprocess will run with full network access."
        )
        return cmd

    def _execute_subprocess(self, working_dir: Path) -> ExecutionResult:
        """Execute code via subprocess (fallback when Docker unavailable)."""
        start_time = time.time()

        # Record files before execution
        pre_files = set(p.name for p in working_dir.iterdir())

        # Symlink only data/tier{1,2}/input/ (not ground_truth/ or prompts/)
        self._provision_input_symlinks(working_dir)

        cmd = ["python", "script.py"]
        if self.network_disabled:
            cmd = self._wrap_with_network_isolation(cmd)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(working_dir),
                capture_output=True,
                timeout=self.timeout_seconds,
                text=True,
            )

            # Collect output files (exclude script.py and data symlink)
            post_files = set(p.name for p in working_dir.iterdir())
            new_files = sorted(post_files - pre_files - {"script.py", "data"})

            execution_time = time.time() - start_time
            return ExecutionResult(
                success=proc.returncode == 0,
                return_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                output_files=new_files,
                execution_time_seconds=execution_time,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Execution timed out after {self.timeout_seconds}s",
                execution_time_seconds=time.time() - start_time,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Subprocess execution error: {e}",
                execution_time_seconds=time.time() - start_time,
            )
