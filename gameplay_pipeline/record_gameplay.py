"""Record gameplay using pysiphon and optionally upload to Hugging Face."""

import argparse
import os
import re
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older versions

from pysiphon import SiphonClient
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from .hf_upload import upload_to_huggingface

console = Console()


def parse_condition_expression(expression: str) -> Dict[str, Any]:
    """
    Parse a condition expression into components.
    
    Supported operators: >, >=, <, <=, ==, !=, &&, ||
    Examples:
        "HeroHp > 0"
        "HeroHp <= 0"
        "HeroHp <= 0 || NpcHp <= 0"  (either condition)
        "HeroHp > 0 && NpcId == 12345"  (both conditions)
    
    Args:
        expression: Condition expression string
        
    Returns:
        Dict with condition details (simple or compound)
    """
    expression = expression.strip()
    
    # Check for logical operators (|| or &&)
    if '||' in expression:
        # OR condition - split and parse each part
        parts = [part.strip() for part in expression.split('||')]
        conditions = [parse_condition_expression(part) for part in parts]
        return {
            "type": "or",
            "conditions": conditions,
        }
    elif '&&' in expression:
        # AND condition - split and parse each part
        parts = [part.strip() for part in expression.split('&&')]
        conditions = [parse_condition_expression(part) for part in parts]
        return {
            "type": "and",
            "conditions": conditions,
        }
    
    # Simple condition: attribute operator value
    pattern = r'^\s*(\w+)\s*(>=|<=|==|!=|>|<)\s*(.+)\s*$'
    match = re.match(pattern, expression)
    
    if not match:
        raise ValueError(f"Invalid condition expression: '{expression}'. Expected format: 'attribute operator value' (e.g., 'HeroHp > 0')")
    
    attribute, operator, value_str = match.groups()
    
    # Parse value
    value_str = value_str.strip()
    try:
        # Try parsing as number
        value = float(value_str)
        if value.is_integer():
            value = int(value)
    except ValueError:
        # Keep as string
        value = value_str
    
    # Map operator to comparison type
    operator_map = {
        '>': 'greater',
        '>=': 'greater_equal',
        '<': 'less',
        '<=': 'less_equal',
        '==': 'equal',
        '!=': 'not_equal',
    }
    
    return {
        "type": operator_map[operator],
        "attribute": attribute,
        "threshold": value,
    }


class GameplayRecorder:
    """Record gameplay sessions using the Siphon service."""

    def __init__(
        self,
        host: str = "localhost:50051",
        config_path: Optional[str] = None,
    ):
        """
        Initialize the gameplay recorder.

        Args:
            host: Siphon server address (e.g., "localhost:50051")
            config_path: Optional path to Siphon config file for initialization
        """
        self.host = host
        self.config_path = config_path
        self.client: Optional[SiphonClient] = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def connect(self):
        """Connect to the Siphon server."""
        console.print(f"[cyan]Connecting to Siphon server at {self.host}...[/cyan]")
        self.client = SiphonClient(self.host)
        self.client.__enter__()

        # Initialize if config provided
        if self.config_path:
            console.print(f"[cyan]Initializing with config: {self.config_path}[/cyan]")
            self.client.init_all(self.config_path)

        # Verify connection
        status = self.client.get_server_status()
        console.print(f"[green]✓ Connected to Siphon server[/green]")
        console.print(f"  Status: {status}")

    def disconnect(self):
        """Disconnect from the Siphon server."""
        if self.client:
            self.client.__exit__(None, None, None)
            console.print("[cyan]Disconnected from Siphon server[/cyan]")

    def test_capture(self, output_path: str = "test_capture.png"):
        """
        Test frame capture functionality.

        Args:
            output_path: Where to save the test capture
        """
        console.print("[cyan]Testing frame capture...[/cyan]")
        self.client.capture_and_save(output_path)
        console.print(f"[green]✓ Test capture saved to {output_path}[/green]")

    def _format_condition(self, condition: Dict[str, Any]) -> str:
        """
        Format a condition dict into a readable string.
        
        Args:
            condition: Condition dictionary
            
        Returns:
            Human-readable condition string
        """
        cond_type = condition["type"]
        
        if cond_type == "or":
            parts = [self._format_condition(c) for c in condition.get("conditions", [])]
            return " || ".join(parts)
        elif cond_type == "and":
            parts = [self._format_condition(c) for c in condition.get("conditions", [])]
            return " && ".join(parts)
        else:
            # Simple condition
            attr = condition.get("attribute", "?")
            thresh = condition.get("threshold", "?")
            
            op_map = {
                "greater": ">",
                "greater_equal": ">=",
                "less": "<",
                "less_equal": "<=",
                "equal": "==",
                "not_equal": "!=",
            }
            op = op_map.get(cond_type, cond_type)
            
            return f"{attr} {op} {thresh}"
    
    def get_attributes(self, attribute_names: list[str]) -> dict:
        """
        Get current values of game attributes.

        Args:
            attribute_names: List of attribute names to retrieve

        Returns:
            Dictionary mapping attribute names to their values
        """
        attributes = {}
        for name in attribute_names:
            try:
                result = self.client.get_attribute(name)
                attributes[name] = result["value"]
            except Exception as e:
                console.print(f"[yellow]Warning: Could not get attribute '{name}': {e}[/yellow]")
                attributes[name] = None
        return attributes

    def check_condition(
        self,
        condition_type: str,
        attribute_name: str = None,
        threshold: Any = None,
        check_interval: float = 0.5,
        conditions: list = None,
    ) -> bool:
        """
        Check if a condition is met based on game attributes.

        Args:
            condition_type: Type of condition ("greater", "greater_equal", "less", "less_equal", "equal", "not_equal", "exists", "and", "or")
            attribute_name: Name of the attribute to check (for simple conditions)
            threshold: Value to compare against (for simple conditions)
            check_interval: Time to wait between checks in seconds
            conditions: List of sub-conditions (for compound conditions)

        Returns:
            True if condition is met, False otherwise
        """
        try:
            # Handle compound conditions (AND/OR)
            if condition_type == "or":
                # OR: return True if ANY condition is met
                if not conditions:
                    return False
                for cond in conditions:
                    if self.check_condition(
                        condition_type=cond["type"],
                        attribute_name=cond.get("attribute"),
                        threshold=cond.get("threshold"),
                        conditions=cond.get("conditions"),
                    ):
                        return True
                return False
            
            elif condition_type == "and":
                # AND: return True if ALL conditions are met
                if not conditions:
                    return True
                for cond in conditions:
                    if not self.check_condition(
                        condition_type=cond["type"],
                        attribute_name=cond.get("attribute"),
                        threshold=cond.get("threshold"),
                        conditions=cond.get("conditions"),
                    ):
                        return False
                return True
            
            # Simple conditions - need attribute_name
            if not attribute_name:
                console.print(f"[yellow]Warning: attribute_name required for condition type {condition_type}[/yellow]")
                return False
            
            result = self.client.get_attribute(attribute_name)
            value = result["value"]
            
            if condition_type == "exists":
                return value is not None
            
            # Convert threshold to match value type for comparison
            if threshold is not None and value is not None:
                try:
                    # If value is numeric, try to convert threshold to numeric
                    if isinstance(value, (int, float)):
                        if isinstance(threshold, str):
                            threshold = float(threshold)
                            if threshold.is_integer():
                                threshold = int(threshold)
                    # If value is string, convert threshold to string
                    elif isinstance(value, str):
                        threshold = str(threshold)
                except (ValueError, TypeError):
                    # If conversion fails, compare as-is
                    pass
            
            if condition_type == "greater":
                return value > threshold
            elif condition_type == "greater_equal":
                return value >= threshold
            elif condition_type == "less":
                return value < threshold
            elif condition_type == "less_equal":
                return value <= threshold
            elif condition_type == "equal":
                return value == threshold
            elif condition_type == "not_equal":
                return value != threshold
            else:
                console.print(f"[yellow]Unknown condition type: {condition_type}[/yellow]")
                return False
        except Exception as e:
            console.print(f"[yellow]Warning: Could not check condition: {e}[/yellow]")
            return False

    def wait_for_condition(
        self,
        condition_dict: Dict[str, Any],
        timeout: Optional[float] = None,
        check_interval: float = 0.5,
    ) -> bool:
        """
        Wait until a condition is met or timeout occurs.

        Args:
            condition_dict: Condition dictionary from parse_condition_expression
            timeout: Maximum time to wait in seconds (None for infinite)
            check_interval: Time between checks in seconds

        Returns:
            True if condition met, False if timeout
        """
        start_time = time.time()
        
        while True:
            if self.check_condition(
                condition_type=condition_dict["type"],
                attribute_name=condition_dict.get("attribute"),
                threshold=condition_dict.get("threshold"),
                conditions=condition_dict.get("conditions"),
            ):
                time.sleep(check_interval)
                return True
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            time.sleep(check_interval)

    def reset_game(self, method: str = "save_reload"):
        """
        Reset the game to start a new episode.
        
        Args:
            method: Reset method to use. Options:
                - "save_reload": Full reset with save file reload 
                - "bonfire": Quick respawn at bonfire/grace 
        
        Override this method to implement additional custom reset logic.
        """
        console.print(f"[cyan]Resetting game (method: {method})...[/cyan]")
        
        if method == "save_reload":
            self._reset_with_save_reload()
        elif method == "bonfire":
            self._reset_at_bonfire()
        else:
            console.print(f"[yellow]Unknown reset method '{method}', using save_reload[/yellow]")
            self._reset_with_save_reload()
    
    def _reset_with_save_reload(self):
        """Full reset with save file reload - for boss fights."""
        console.print("[dim]Using save reload method...[/dim]")
        
        # Exit to menu
        self.client.input_key_tap(["ESC"])
        time.sleep(0.3)
        
        # Navigate menu (UP_ARROW, E)
        self.client.input_key_tap(["UP_ARROW"])
        time.sleep(0.3)
        self.client.input_key_tap(["E"])
        time.sleep(0.3)
        
        # Confirm quit (Z, E, LEFT_ARROW, E)
        self.client.input_key_tap(["Z"])
        time.sleep(0.3)
        self.client.input_key_tap(["E"])
        time.sleep(0.3)
        self.client.input_key_tap(["LEFT_ARROW"])
        time.sleep(0.3)
        self.client.input_key_tap(["E"])
        time.sleep(12.0)
        time.sleep(5.0)

        console.print("[green]✓ Game quit[/green]")
        # Copy save file if configured
        if hasattr(self, 'save_file_name') and hasattr(self, 'save_file_dir') and self.save_file_name and self.save_file_dir:
            console.print(f"[yellow]⟳ Copying save file: {self.save_file_name}[/yellow]")
            try:
                import os
                source_path = os.path.join(self.save_file_dir, self.save_file_name)
                dest_path = os.path.join(self.save_file_dir, "ER0000.sl2")
                
                # PowerShell command to copy the save file
                result = self.client.execute_command(
                    "powershell",
                    args=[
                        "-Command",
                        f'Copy-Item -Path "{source_path}" -Destination "{dest_path}" -Force'
                    ],
                    timeout_seconds=10,
                    capture_output=True
                )
                console.print("[green]✓ Save file copied[/green]")
                time.sleep(2.0)
            except Exception as e:
                console.print(f"[red]✗ Failed to copy save file: {e}[/red]")
        else:
            console.print("[yellow]Waiting for save file copy (12s)...[/yellow]")
        
        # Re-enter game
        self.client.input_key_tap(["E"])
        time.sleep(2.0)
        self.client.input_key_tap(["E"])

        time.sleep(8.0)  # Wait for game to load
        
        console.print("[green]✓ Save reload complete[/green]")
    
    def _reset_at_bonfire(self):
        """Quick respawn at bonfire/Site of Grace - for exploration/practice."""
        console.print("[dim]Waiting for bonfire respawn...[/dim]")
        
        time.sleep(20.0)  # Wait for respawn
        console.print("[green]✓ Bonfire respawn complete[/green]")

    def record_session(
        self,
        attribute_names: list[str],
        output_directory: str = "./recordings",
        max_duration_seconds: int = 60,
        show_preview: bool = True,
    ) -> dict:
        """
        Record a gameplay session with automatic frame and attribute capture.

        Args:
            attribute_names: List of game attributes to record (e.g., ["health", "mana", "position"])
            output_directory: Directory where recording will be saved
            max_duration_seconds: Maximum recording duration in seconds
            show_preview: Whether to show live preview of attributes

        Returns:
            Dictionary with recording statistics and file path
        """
        # Create output directory
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print("\n[bold cyan]Starting gameplay recording...[/bold cyan]")
        console.print(f"  Attributes: {', '.join(attribute_names)}")
        console.print(f"  Max duration: {max_duration_seconds}s")
        console.print(f"  Output: {output_directory}")

        # Start recording session
        result = self.client.start_recording(
            attribute_names=attribute_names,
            output_directory=str(output_dir),
            max_duration_seconds=max_duration_seconds,
        )

        session_id = result["session_id"]
        console.print(f"[green]✓ Recording started (Session ID: {session_id})[/green]")
        console.print("[yellow]Press Ctrl+C to stop recording[/yellow]\n")

        try:
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Recording...", total=max_duration_seconds)

                start_time = time.time()
                last_status_time = start_time

                while True:
                    elapsed = time.time() - start_time

                    # Update progress
                    progress.update(task, completed=min(elapsed, max_duration_seconds))

                    # Show status every second
                    if show_preview and time.time() - last_status_time >= 1.0:
                        status = self.client.get_recording_status(session_id)
                        current_frame = status.get("current_frame", 0)
                        
                        # Get current attribute values for preview
                        attrs = self.get_attributes(attribute_names[:3])  # Show first 3
                        attr_str = " | ".join([f"{k}: {v}" for k, v in attrs.items()])
                        
                        progress.console.print(
                            f"[dim]Frame {current_frame} | {attr_str}[/dim]",
                            end="\r",
                        )
                        last_status_time = time.time()

                    # Check if max duration reached
                    if elapsed >= max_duration_seconds:
                        console.print("\n[yellow]Max duration reached[/yellow]")
                        break

                    # Small sleep to avoid busy loop
                    time.sleep(0.1)

        except KeyboardInterrupt:
            console.print("\n[yellow]Recording interrupted by user[/yellow]")

        # Stop recording
        console.print("\n[cyan]Stopping recording...[/cyan]")
        stats = self.client.stop_recording(session_id)

        console.print("[green]✓ Recording stopped[/green]")
        console.print(f"  Total frames: {stats.get('total_frames', 0)}")
        console.print(f"  Duration: {stats.get('duration_seconds', stats.get('duration', 0)):.2f}s")
        console.print(f"  Actual FPS: {stats.get('actual_fps', stats.get('fps', 0)):.2f}")

        # Download recording (receives multiple files: .h5, .mp4, etc.)
        # Create session-specific subdirectory
        session_output_dir = output_dir / session_id
        session_output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Downloading recording files to {session_output_dir}...[/cyan]")
        self.client.download_recording(session_id, str(session_output_dir))
        console.print(f"[green]✓ Recording files saved to {session_output_dir}/[/green]")

        return {
            "session_id": session_id,
            "output_directory": str(session_output_dir),
            "stats": stats,
        }

    def record_endless(
        self,
        attribute_names: list[str],
        output_directory: str = "./recordings",
        max_episode_duration: int = 300,
        max_episodes: Optional[int] = None,
        start_condition: Optional[Dict[str, Any]] = None,
        stop_conditions: Optional[list[Dict[str, Any]]] = None,
        upload_config: Optional[Dict[str, Any]] = None,
        save_file_name: Optional[str] = None,
        save_file_dir: Optional[str] = None,
        show_preview: bool = True,
    ) -> list[dict]:
        """
        Record gameplay episodes in endless mode with automatic start/stop/reset.

        Args:
            attribute_names: List of game attributes to record
            output_directory: Directory where recordings will be saved
            max_episode_duration: Maximum duration per episode in seconds
            max_episodes: Maximum number of episodes to record (None for infinite)
            start_condition: Dict with keys: type, attribute, threshold
            stop_conditions: List of dicts with keys: condition, reset_method (OR logic between them)
            upload_config: Dict with keys: enabled, repo_id, token
            save_file_name: Name of the backup save file to copy (e.g., "margit_save.sl2")
            save_file_dir: Directory containing save files
            show_preview: Whether to show live preview

        Returns:
            List of recording results for each episode
        """
        self.save_file_name = save_file_name
        self.save_file_dir = save_file_dir
        # Set defaults
        start_cond = start_condition or {"type": "greater", "attribute": "HeroHp", "threshold": 0}
        stop_conds = stop_conditions or [
            {"condition": {"type": "equal", "attribute": "HeroHp", "threshold": 0}, "reset_method": "save_reload"}
        ]
        upload_cfg = upload_config or {"enabled": False}

        results = []
        
        # Create output directory
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Count existing recordings to determine starting episode number for filenames only
        existing_files = list(output_dir.glob("*.h5"))
        episode_file_counter = len(existing_files)

        console.print("\n[bold cyan]Starting endless recording mode...[/bold cyan]")
        console.print(f"  Max episodes: {max_episodes or 'unlimited'}")
        console.print(f"  Episode duration: {max_episode_duration}s")
        if start_cond:
            console.print(f"  Start condition: {self._format_condition(start_cond)}")
        console.print(f"  Stop conditions ({len(stop_conds)}):")
        for i, sc in enumerate(stop_conds, 1):
            cond_str = self._format_condition(sc["condition"])
            reset_str = sc["reset_method"]
            console.print(f"    {i}. {cond_str} → reset: {reset_str}")
        console.print("[yellow]Press Ctrl+C to stop endless recording[/yellow]\n")

        try:
            while max_episodes is None or len(results) < max_episodes:
                console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
                console.print(f"[bold magenta]Recording Episode (#{len(results) + 1} this session)[/bold magenta]")
                console.print(f"[bold magenta]{'='*60}[/bold magenta]\n")

                # Wait for start condition (if specified)
                if start_cond:
                    # Pretty print the condition
                    cond_str = self._format_condition(start_cond)
                    console.print(f"[cyan]Waiting for start condition: {cond_str}...[/cyan]")
                    if not self.wait_for_condition(
                        condition_dict=start_cond,
                        timeout=None,
                        check_interval=0.5,
                    ):
                        console.print("[yellow]Start condition timeout[/yellow]")
                        continue
                    console.print("[green]✓ Start condition met, beginning recording[/green]")
                else:
                    console.print("[green]No start condition - starting immediately[/green]")

                # Start recording
                result = self.client.start_recording(
                    attribute_names=attribute_names,
                    output_directory=str(output_dir),
                    max_duration_seconds=max_episode_duration,
                )

                session_id = result["session_id"]
                console.print(f"[green]✓ Recording started (Session: {session_id})[/green]")

                # Monitor for stop condition or max duration
                start_time = time.time()
                last_status_time = start_time

                # Track which stop condition was met and its reset method
                met_reset_method = None
                met_condition_index = None
                
                try:
                    while True:
                        elapsed = time.time() - start_time

                        # Check all stop conditions (OR logic)
                        for idx, stop_cond_info in enumerate(stop_conds):
                            cond = stop_cond_info["condition"]
                            if self.check_condition(
                                condition_type=cond["type"],
                                attribute_name=cond.get("attribute"),
                                threshold=cond.get("threshold"),
                                conditions=cond.get("conditions"),
                            ):
                                cond_str = self._format_condition(cond)
                                met_reset_method = stop_cond_info["reset_method"]
                                met_condition_index = idx
                                console.print(f"\n[yellow]Stop condition {idx} met: {cond_str} (reset: {met_reset_method})[/yellow]")
                                break
                        
                        # Exit if any stop condition was met
                        if met_reset_method:
                            break

                        # Check max duration
                        if elapsed >= max_episode_duration:
                            console.print("\n[yellow]Max episode duration reached[/yellow]")
                            # Use first reset method if timeout
                            met_reset_method = stop_conds[0]["reset_method"] if stop_conds else "save_reload"
                            met_condition_index = 0 if stop_conds else None
                            break

                        # Show preview
                        if show_preview and time.time() - last_status_time >= 1.0:
                            status = self.client.get_recording_status(session_id)
                            current_frame = status.get("current_frame", 0)
                            
                            attrs = self.get_attributes(attribute_names[:3])
                            attr_str = " | ".join([f"{k}: {v}" for k, v in attrs.items()])
                            
                            console.print(
                                f"[dim]Frame {current_frame} | {attr_str} | {elapsed:.1f}s[/dim]",
                                end="\r",
                            )
                            last_status_time = time.time()

                        time.sleep(0.1)

                except KeyboardInterrupt:
                    console.print("\n[yellow]Episode interrupted[/yellow]")
                    raise  # Re-raise to exit outer loop

                # Stop recording
                console.print("\n[cyan]Stopping episode recording...[/cyan]")
                stats = self.client.stop_recording(session_id)

                # Debug: print available keys
                console.print(f"[dim]Stats keys: {list(stats.keys())}[/dim]")
                console.print(f"[dim]Stats content: {stats}[/dim]")

                console.print("[green]✓ Episode recording stopped[/green]")
                console.print(f"  Frames: {stats.get('total_frames', 0)}")
                console.print(f"  Duration: {stats.get('duration_seconds', stats.get('duration', 0)):.2f}s")
                console.print(f"  FPS: {stats.get('actual_fps', stats.get('fps', 0)):.2f}")

                # Download recording (receives multiple files: .h5, .mp4, etc.)
                episode_file_counter += 1
                timestamp = int(time.time() * 1000)  # millisecond precision

                # Create session-specific subdirectory
                session_output_dir = output_dir / session_id
                session_output_dir.mkdir(parents=True, exist_ok=True)

                console.print(f"[cyan]Downloading episode files to {session_output_dir}/ (session: {session_id})...[/cyan]")
                self.client.download_recording(session_id, str(session_output_dir))
                console.print(f"[green]✓ Episode files saved to {session_output_dir}/[/green]")
                
                # Note: Files are saved with server-generated names (e.g., session_id.h5, session_id.mp4)
                # The condition index and episode counter are tracked here for reference
                cond_suffix = f"_cond{met_condition_index}" if met_condition_index is not None else ""

                episode_result = {
                    "session_id": session_id,
                    "output_directory": str(session_output_dir),
                    "episode_number": episode_file_counter,
                    "condition_suffix": cond_suffix,
                    "stats": stats,
                    "stop_condition_index": met_condition_index,
                    "timestamp": timestamp,
                }
                results.append(episode_result)

                # Upload if enabled
                if upload_cfg.get("enabled") and upload_cfg.get("repo_id"):
                    try:
                        console.print("[cyan]Uploading episode files to Hugging Face...[/cyan]")
                        
                        # Find all files in the session directory
                        session_files = list(session_output_dir.glob("*"))
                        
                        upload_urls = []
                        for file_path in session_files:
                            if file_path.is_file():
                                console.print(f"[dim]Uploading {file_path.name}...[/dim]")
                                url = upload_to_huggingface(
                                    file_path=str(file_path),
                                    repo_id=upload_cfg["repo_id"],
                                    repo_type="dataset",
                                    token=upload_cfg.get("token"),
                                )
                                upload_urls.append(url)
                        
                        console.print(f"[green]✓ Uploaded {len(upload_urls)} files[/green]")
                        episode_result["upload_urls"] = upload_urls
                    except Exception as e:
                        console.print(f"[red]Upload failed: {e}[/red]")

                # Reset game after episode using the appropriate reset method
                # (even on last episode to ensure consistent state)
                self.reset_game(method=met_reset_method or "save_reload")

        except KeyboardInterrupt:
            console.print("\n[yellow]Endless recording stopped by user[/yellow]")

        console.print(f"\n[bold green]✓ Completed {len(results)} episodes[/bold green]")
        return results


def load_recording_config(config_path: str, episode_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load recording configuration from TOML file.

    Args:
        config_path: Path to the TOML config file
        episode_name: Name of the episode to load settings for

    Returns:
        Dictionary with configuration settings
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Get episode-specific settings if provided
    episode_config = None
    if episode_name:
        if "episodes" not in config or episode_name not in config["episodes"]:
            raise ValueError(f"Episode '{episode_name}' not found in config file")
        episode_config = config["episodes"][episode_name]

    # Build combined config
    result = {
        "server": config.get("server", {}),
        "recording": config.get("recording", {}),
        "upload": config.get("upload", {}),
    }

    # Add episode-specific settings
    if episode_config:
        # Parse start condition
        start_cond = episode_config.get("start_condition")
        if isinstance(start_cond, str):
            start_cond = parse_condition_expression(start_cond)
        elif isinstance(start_cond, dict):
            pass  # Backward compatibility
        
        # Parse stop conditions (new format with reset methods)
        stop_conditions_config = episode_config.get("stop_conditions")
        stop_conditions = []
        
        if stop_conditions_config:
            # New format: list of {condition, reset_method}
            if isinstance(stop_conditions_config, list):
                for sc in stop_conditions_config:
                    if isinstance(sc, dict):
                        cond_expr = sc.get("condition")
                        reset_method = sc.get("reset_method", "save_reload")
                        if isinstance(cond_expr, str):
                            parsed_cond = parse_condition_expression(cond_expr)
                        else:
                            parsed_cond = cond_expr
                        stop_conditions.append({
                            "condition": parsed_cond,
                            "reset_method": reset_method,
                        })
        else:
            # Backward compatibility: single stop_condition
            stop_cond = episode_config.get("stop_condition")
            reset_method = episode_config.get("reset_method", "save_reload")
            if stop_cond:
                if isinstance(stop_cond, str):
                    stop_cond = parse_condition_expression(stop_cond)
                stop_conditions.append({
                    "condition": stop_cond,
                    "reset_method": reset_method,
                })
        
        result["episode"] = {
            "name": episode_config.get("name", episode_name),
            "description": episode_config.get("description", ""),
            "attributes": episode_config.get("attributes", []),
            "start_condition": start_cond,
            "stop_conditions": stop_conditions,
            "save_file_name": episode_config.get("save_file_name"),
            "save_file_dir": episode_config.get("save_file_dir"),
        }

    return result


def main():
    """CLI entry point for gameplay recording."""
    parser = argparse.ArgumentParser(
        description="Record gameplay using Siphon service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file mode (new simplified approach)
    parser.add_argument(
        "--recording-config",
        help="Path to recording config TOML file",
    )
    parser.add_argument(
        "--episode",
        help="Episode name from config file (e.g., 'margit', 'godrick')",
    )

    # Connection settings
    parser.add_argument(
        "--host",
        default="localhost:50051",
        help="Siphon server address",
    )
    parser.add_argument(
        "--siphon-config",
        help="Path to Siphon config file (e.g., elden_ring.toml) - REQUIRED",
    )

    # Legacy recording settings (for manual mode)
    parser.add_argument(
        "--attributes",
        help="Comma-separated list of attributes to record (e.g., health,mana,position)",
    )
    parser.add_argument(
        "--output-dir",
        default="./recordings",
        help="Output directory for recordings",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Maximum recording duration in seconds",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview of attributes",
    )

    # Test mode
    parser.add_argument(
        "--test-capture",
        action="store_true",
        help="Test frame capture and exit",
    )

    # Upload settings
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload recording to Hugging Face after completion",
    )
    parser.add_argument(
        "--repo-id",
        help="Hugging Face repository ID (e.g., username/repo-name)",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )

    # Episode settings
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Maximum number of episodes to record (default: unlimited)",
    )
    parser.add_argument(
        "--start-condition",
        help="Start condition expression (e.g., 'HeroHp > 0'). If not specified, starts immediately.",
    )
    parser.add_argument(
        "--stop-condition",
        help="Stop condition expression (e.g., 'HeroHp <= 0'). If not specified, runs until interrupted.",
    )

    args = parser.parse_args()

    # Validate siphon config is provided
    if not args.siphon_config:
        console.print("[red]Error: --siphon-config is required[/red]")
        console.print("[yellow]Example: --siphon-config gameplay_pipeline/elden_ring.toml[/yellow]")
        return 1

    # Determine mode: config file or manual
    if args.recording_config:
        # Config file mode
        try:
            config = load_recording_config(args.recording_config, args.episode)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            return 1

        # Extract settings from config
        host = config["server"].get("host", "localhost:50051")
        output_dir = config["recording"].get("output_directory", "./recordings")
        max_duration = config["recording"].get("max_episode_duration", 300)
        show_preview = config["recording"].get("show_preview", True)
        
        # Get HF token from env if not in args
        hf_token = args.hf_token or os.getenv("HF_TOKEN")
        
        # Upload settings
        upload_enabled = args.upload or config["upload"].get("enabled", False)
        repo_id = args.repo_id or config["upload"].get("repo_id")
        
        upload_cfg = {
            "enabled": upload_enabled,
            "repo_id": repo_id,
            "token": hf_token,
        }

        # Episode-specific settings
        if "episode" in config:
            attribute_names = config["episode"]["attributes"]
            start_cond = config["episode"].get("start_condition")
            stop_conds = config["episode"].get("stop_conditions", [])
            save_file_name = config["episode"].get("save_file_name")
            save_file_dir = config["episode"].get("save_file_dir")
            episode_name = config["episode"]["name"]
            
            console.print(f"[bold cyan]Recording Episode: {episode_name}[/bold cyan]")
            if config["episode"].get("description"):
                console.print(f"  {config['episode']['description']}")
        else:
            console.print("[red]Error: No episode specified in config mode[/red]")
            console.print("[yellow]Use --episode <name> to select an episode from the config[/yellow]")
            return 1

    else:
        # Manual/legacy mode
        if not args.attributes:
            console.print("[red]Error: --attributes required in manual mode[/red]")
            return 1
            
        attribute_names = [a.strip() for a in args.attributes.split(",")]
        host = args.host
        output_dir = args.output_dir
        max_duration = args.duration
        show_preview = not args.no_preview
        save_file_name = None  # Not available in manual mode
        save_file_dir = None
        
        # Parse condition expressions
        start_cond = None
        stop_conds = []
        
        if args.start_condition:
            try:
                start_cond = parse_condition_expression(args.start_condition)
            except ValueError as e:
                console.print(f"[red]Error parsing start condition: {e}[/red]")
                return 1
        
        if args.stop_condition:
            try:
                parsed_stop = parse_condition_expression(args.stop_condition)
                stop_conds = [{
                    "condition": parsed_stop,
                    "reset_method": "save_reload"  # Default for manual mode
                }]
            except ValueError as e:
                console.print(f"[red]Error parsing stop condition: {e}[/red]")
                return 1
        
        hf_token = args.hf_token or os.getenv("HF_TOKEN")
        upload_cfg = {
            "enabled": args.upload,
            "repo_id": args.repo_id,
            "token": hf_token,
        }

    try:
        with GameplayRecorder(host=host, config_path=args.siphon_config) as recorder:
            if args.test_capture:
                # Test mode
                recorder.test_capture()
                console.print("[green]✓ Test completed successfully[/green]")
            else:
                # Episode recording mode (default)
                if upload_cfg.get("enabled") and not upload_cfg.get("repo_id"):
                    console.print("[red]Error: --repo-id required for upload[/red]")
                    return 1
                
                results = recorder.record_endless(
                    attribute_names=attribute_names,
                    output_directory=output_dir,
                    max_episode_duration=max_duration,
                    max_episodes=args.max_episodes,
                    start_condition=start_cond,
                    stop_conditions=stop_conds,
                    upload_config=upload_cfg,
                    save_file_name=save_file_name,
                    save_file_dir=save_file_dir,
                    show_preview=show_preview,
                )
                
                console.print(f"\n[bold green]✓ All done! Recorded {len(results)} episodes[/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

