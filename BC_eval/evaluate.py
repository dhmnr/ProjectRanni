"""Main evaluation script for BC models in EldenGym."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel

from .model_loader import load_model, LoadedModel
from .agents.random import RandomAgent
from .agents.base import BaseAgent
from .agents.temporal import TemporalAgent

console = Console()
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def create_agent(
    config: Dict,
    model: Optional[LoadedModel] = None,
    env=None,
) -> Union[RandomAgent, BaseAgent, TemporalAgent]:
    """Create agent based on config.

    Args:
        config: Evaluation config dict
        model: Optional loaded model (not needed for random agent)
        env: Optional environment instance for action space mapping

    Returns:
        Agent instance
    """
    agent_type = config.get("agent", {}).get("type", "model")
    action_threshold = config.get("evaluation", {}).get("action_threshold", 0.5)

    # Get frame shape from config or use default
    frame_shape = tuple(
        config.get("evaluation", {}).get("frame_shape", [3, 144, 256])
    )

    # Extract action mapping info from environment if available
    env_action_keys = None
    env_keybinds = None
    if env is not None:
        env_action_keys = getattr(env, 'action_keys', None)
        env_keybinds = getattr(env, 'keybinds', None)
        if env_action_keys and env_keybinds:
            console.print(f"[cyan]Environment has {len(env_action_keys)} actions[/cyan]")
            console.print(f"[cyan]Action keys: {env_action_keys}[/cyan]")

    if agent_type == "random":
        # For random agent, use environment's action space size
        if env is not None:
            num_actions = env.action_space.shape[0]
        else:
            num_actions = config.get("agent", {}).get("num_actions", 7)
        action_prob = config.get("agent", {}).get("action_prob", 0.1)
        seed = config.get("agent", {}).get("seed", None)

        return RandomAgent(
            num_actions=num_actions,
            action_prob=action_prob,
            seed=seed,
        )

    elif agent_type == "model":
        if model is None:
            raise ValueError("Model required for 'model' agent type")

        if model.is_temporal:
            return TemporalAgent(
                model=model,
                action_threshold=action_threshold,
                frame_shape=frame_shape,
                env_action_keys=env_action_keys,
                env_keybinds=env_keybinds,
            )
        else:
            return BaseAgent(
                model=model,
                action_threshold=action_threshold,
                frame_shape=frame_shape,
                env_action_keys=env_action_keys,
                env_keybinds=env_keybinds,
            )

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_environment(config: Dict):
    """Create EldenGym environment from config.

    Args:
        config: Evaluation config dict

    Returns:
        EldenGym environment instance
    """
    try:
        import eldengym
    except ImportError:
        raise ImportError(
            "eldengym not installed. Install with: pip install eldengym"
        )

    env_config = config.get("environment", {})

    # Environment ID for eldengym.make()
    env_id = env_config.get("env_id", "Margit-v0")

    # Optional parameters to pass to make()
    host = env_config.get("host", "localhost:50051")
    launch_game = env_config.get("launch_game", False)
    max_steps = env_config.get("max_steps", None)
    save_file_name = env_config.get("save_file_name", None)
    save_file_dir = env_config.get("save_file_dir", None)

    console.print(f"[cyan]Creating EldenGym environment...[/cyan]")
    console.print(f"  Environment ID: {env_id}")
    console.print(f"  Host: {host}")
    console.print(f"  Launch game: {launch_game}")
    

    # Build kwargs for make()
    make_kwargs = {
        "host": host,
        "launch_game": launch_game,
        "save_file_name": save_file_name,
        "save_file_dir": save_file_dir,
    }
    if max_steps is not None:
        make_kwargs["max_steps"] = max_steps

    env = eldengym.make(env_id, **make_kwargs)

    console.print("[green]✓ Environment created[/green]")
    return env


def run_episode(
    env,
    agent: Union[RandomAgent, BaseAgent, TemporalAgent],
    render: bool = False,
    max_steps: Optional[int] = None,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single evaluation episode.

    Args:
        env: EldenGym environment
        agent: Agent to evaluate
        render: Whether to render frames
        max_steps: Maximum steps per episode (overrides env setting)
        log_file: Path to write detailed step-by-step log (CSV format)

    Returns:
        Episode metrics dict
    """
    # Reset environment and agent
    obs, info = env.reset()
    agent.reset()

    episode_reward = 0.0
    episode_steps = 0
    start_time = time.time()

    # Track metrics using normalized HP from info (eldengym convention)
    # info contains: player_hp_normalized, boss_hp_normalized (0.0 to 1.0)
    initial_player_hp_norm = info.get("player_hp_normalized", 1.0)
    initial_boss_hp_norm = info.get("boss_hp_normalized", 1.0)
    min_boss_hp_norm = initial_boss_hp_norm

    # Get action key names for logging
    env_action_keys = getattr(env, 'action_keys', None)
    env_keybinds = getattr(env, 'keybinds', None)

    # Setup log file
    log_handle = None
    samples_dir = None
    if log_file:
        log_handle = open(log_file, 'w')
        # Write CSV header - include raw probabilities for debugging
        log_handle.write("step,timestamp_ms,player_hp,boss_hp,reward,inference_ms,step_ms,active_keys,active_actions,probs\n")
        console.print(f"[yellow]Logging to: {log_file}[/yellow]")

        # Create samples directory for saving random frames
        samples_dir = Path(log_file).parent / "samples" / Path(log_file).stem
        samples_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]Saving samples to: {samples_dir}[/yellow]")

    done = False
    terminated = False
    truncated = False

    # Get action threshold from agent
    action_threshold = getattr(agent, 'action_threshold', 0.5)

    while not done:
        step_start = time.time()
        timestamp_ms = (step_start - start_time) * 1000

        # Get action from agent
        # Combine observation with info for state-based agents
        # obs is a dict with 'frame' and memory attributes from eldengym
        obs_dict = {"frame": obs, **info} if isinstance(obs, np.ndarray) else {**obs, **info}

        inference_start = time.time()
        # Get action (probs stored in agent._last_probs for logging)
        action = agent.act(obs_dict)
        inference_ms = (time.time() - inference_start) * 1000

        # Get raw probs for logging (re-threshold to get model probs, not env action)
        probs = getattr(agent, '_last_probs', None)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_ms = (time.time() - step_start) * 1000

        episode_reward += reward
        episode_steps += 1

        # Track boss HP (normalized)
        boss_hp_norm = info.get("boss_hp_normalized", min_boss_hp_norm)
        min_boss_hp_norm = min(min_boss_hp_norm, boss_hp_norm)
        player_hp_norm = info.get("player_hp_normalized", 1.0)

        # Log to file
        if log_handle:
            active_indices = np.where(action > 0)[0]
            if env_action_keys:
                active_keys = [env_action_keys[i] for i in active_indices]
                # Also get semantic action names
                active_actions = [env_keybinds.get(k, k) for k in active_keys] if env_keybinds else active_keys
            else:
                active_keys = [str(i) for i in active_indices]
                active_actions = active_keys

            # Format probs as compact string
            probs_str = ""
            if probs is not None:
                probs_str = ";".join([f"{p:.3f}" for p in probs])

            log_handle.write(
                f"{episode_steps},"
                f"{timestamp_ms:.1f},"
                f"{player_hp_norm:.4f},"
                f"{boss_hp_norm:.4f},"
                f"{reward:.4f},"
                f"{inference_ms:.2f},"
                f"{step_ms:.2f},"
                f"\"{'+'.join(active_keys) if active_keys else 'none'}\","
                f"\"{'+'.join(active_actions) if active_actions else 'none'}\","
                f"\"{probs_str}\"\n"
            )
            log_handle.flush()

            # Randomly save samples (5% chance) for debugging
            if samples_dir and np.random.random() < 0.05:
                import cv2
                import json

                # Get raw frame from observation
                raw_frame = obs_dict.get("frame")
                if raw_frame is not None and isinstance(raw_frame, np.ndarray):
                    # Save raw frame
                    frame_path = samples_dir / f"step_{episode_steps:04d}_raw.png"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))

                    # Save metadata
                    meta = {
                        "step": episode_steps,
                        "player_hp": float(player_hp_norm),
                        "boss_hp": float(boss_hp_norm),
                        "probs": [float(p) for p in probs] if probs is not None else [],
                        "active_actions": active_actions,
                        "raw_frame_shape": list(raw_frame.shape),
                    }
                    meta_path = samples_dir / f"step_{episode_steps:04d}_meta.json"
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)

        # Render if requested
        if render:
            env.render()

        # Check max steps
        if max_steps and episode_steps >= max_steps:
            truncated = True
            break

    episode_time = time.time() - start_time

    # Close log file and print summary
    if log_handle:
        log_handle.close()
        console.print(f"[green]✓ Log saved: {log_file}[/green]")
        console.print(f"  Total steps: {episode_steps}")
        console.print(f"  Total time: {episode_time:.2f}s")
        console.print(f"  Avg FPS: {episode_steps / episode_time:.1f}")

    # Calculate metrics using normalized values
    player_hp_norm_remaining = info.get("player_hp_normalized", 0.0)
    boss_hp_norm_remaining = info.get("boss_hp_normalized", 0.0)

    # Damage dealt to boss (as percentage of initial HP)
    damage_dealt_pct = (initial_boss_hp_norm - min_boss_hp_norm) * 100

    # Survival (player HP remaining as percentage)
    survival_pct = player_hp_norm_remaining * 100

    return {
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "episode_time": episode_time,
        "damage_dealt_pct": damage_dealt_pct,
        "survival_pct": survival_pct,
        "target_hp_remaining": boss_hp_norm_remaining,
        "player_hp_remaining": player_hp_norm_remaining,
        "terminated": terminated,
        "truncated": truncated,
        "victory": boss_hp_norm_remaining <= 0,
    }


def evaluate(config_path: str, overrides: Optional[Dict] = None):
    """Main evaluation function.

    Args:
        config_path: Path to evaluation config YAML
        overrides: Optional dict of config overrides
    """
    # Load config
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    # Apply overrides
    if overrides:
        if "checkpoint_path" in overrides:
            config.setdefault("checkpoint", {})["path"] = overrides["checkpoint_path"]
        if "num_episodes" in overrides:
            config.setdefault("evaluation", {})["num_episodes"] = overrides["num_episodes"]
        if "render" in overrides:
            config.setdefault("evaluation", {})["render"] = overrides["render"]

    console.print(Panel(
        OmegaConf.to_yaml(OmegaConf.create(config)),
        title="[bold]Evaluation Configuration[/bold]",
        expand=False,
    ))

    # Determine agent type
    agent_type = config.get("agent", {}).get("type", "model")

    # Create environment first (needed for action space mapping)
    env = create_environment(config)

    # Load model if needed
    model = None
    if agent_type == "model":
        checkpoint_config = config.get("checkpoint", {})
        checkpoint_path = checkpoint_config.get("path")
        training_config_path = checkpoint_config.get("training_config")
        anim_mappings_path = checkpoint_config.get("anim_mappings_path")

        if not checkpoint_path:
            raise ValueError("checkpoint.path is required for model agent")
        if not training_config_path:
            raise ValueError("checkpoint.training_config is required for model agent")

        console.print(f"[cyan]Loading model from {checkpoint_path}[/cyan]")
        model = load_model(
            checkpoint_path=checkpoint_path,
            training_config_path=training_config_path,
            anim_mappings_path=anim_mappings_path,
        )
        console.print(f"[green]✓ Model loaded: {model.model_name}[/green]")

    # Create agent with environment for action mapping
    agent = create_agent(config, model, env=env)
    console.print(f"[green]✓ Agent created: {agent}[/green]")

    # Evaluation settings
    eval_config = config.get("evaluation", {})
    num_episodes = eval_config.get("num_episodes", 10)
    render = eval_config.get("render", False)
    max_steps = eval_config.get("max_steps_per_episode", None)

    # Initialize wandb if enabled
    logging_config = config.get("logging", {})
    use_wandb = logging_config.get("use_wandb", False)
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project=logging_config.get("wandb_project", "ProjectRanni-Eval"),
                entity=logging_config.get("wandb_entity", None),
                name=logging_config.get("wandb_run_name", None),
                config=config,
            )
            console.print("[green]✓ Initialized WandB[/green]")
        except ImportError:
            console.print("[yellow]Warning: wandb not installed, skipping logging[/yellow]")
            use_wandb = False

    # Run evaluation episodes
    console.print(f"\n[bold cyan]Running {num_episodes} evaluation episodes...[/bold cyan]")

    # Setup log directory
    log_dir = Path("BC_eval/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    all_metrics: List[Dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Episodes", total=num_episodes)

        for ep_idx in range(num_episodes):
            progress.update(task, description=f"[cyan]Episode {ep_idx + 1}/{num_episodes}")

            # Create log file for this episode
            log_file = log_dir / f"episode_{timestamp}_ep{ep_idx+1:02d}.csv"

            try:
                metrics = run_episode(
                    env=env,
                    agent=agent,
                    render=render,
                    max_steps=max_steps,
                    log_file=str(log_file),
                )
                metrics["episode"] = ep_idx + 1
                metrics["log_file"] = str(log_file)
                all_metrics.append(metrics)

                # Log to wandb
                if use_wandb:
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()})

                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"[cyan]Ep {ep_idx + 1} │ "
                        f"Dmg: {metrics['damage_dealt_pct']:.1f}% │ "
                        f"Steps: {metrics['episode_steps']}"
                    ),
                )

            except Exception as e:
                console.print(f"[red]Episode {ep_idx + 1} failed: {e}[/red]")
                logger.exception(f"Episode {ep_idx + 1} failed")
                progress.update(task, advance=1)

    # Close environment
    env.close()

    # Compute aggregate metrics
    if all_metrics:
        aggregate = compute_aggregate_metrics(all_metrics)
        print_results(all_metrics, aggregate, agent)

        # Log final metrics to wandb
        if use_wandb:
            wandb.log({f"eval/final_{k}": v for k, v in aggregate.items()})
            wandb.finish()
    else:
        console.print("[red]No episodes completed successfully[/red]")


def compute_aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
    """Compute aggregate statistics from episode metrics.

    Args:
        metrics_list: List of episode metric dicts

    Returns:
        Dict with aggregate statistics
    """
    if not metrics_list:
        return {}

    # Extract metric arrays
    rewards = [m["episode_reward"] for m in metrics_list]
    steps = [m["episode_steps"] for m in metrics_list]
    damage = [m["damage_dealt_pct"] for m in metrics_list]
    survival = [m["survival_pct"] for m in metrics_list]
    victories = [m["victory"] for m in metrics_list]

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_steps": np.mean(steps),
        "std_steps": np.std(steps),
        "mean_damage_pct": np.mean(damage),
        "std_damage_pct": np.std(damage),
        "max_damage_pct": np.max(damage),
        "mean_survival_pct": np.mean(survival),
        "victory_rate": np.mean(victories) * 100,
        "num_episodes": len(metrics_list),
    }


def print_results(
    metrics_list: List[Dict],
    aggregate: Dict[str, float],
    agent: Union[RandomAgent, BaseAgent, TemporalAgent],
):
    """Print evaluation results in formatted tables.

    Args:
        metrics_list: List of episode metric dicts
        aggregate: Aggregate statistics
        agent: The agent that was evaluated
    """
    console.print("\n")

    # Episode details table
    table = Table(title="Episode Results", show_header=True, header_style="bold magenta")
    table.add_column("Episode", justify="right", style="cyan")
    table.add_column("Reward", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Damage %", justify="right")
    table.add_column("Survival %", justify="right")
    table.add_column("Result", justify="center")

    for m in metrics_list:
        result = "🏆" if m["victory"] else ("⏱️" if m["truncated"] else "💀")
        table.add_row(
            str(m["episode"]),
            f"{m['episode_reward']:.2f}",
            str(m["episode_steps"]),
            f"{m['damage_dealt_pct']:.1f}%",
            f"{m['survival_pct']:.1f}%",
            result,
        )

    console.print(table)

    # Summary table
    summary = Table(title="Evaluation Summary", show_header=True, header_style="bold green")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")

    summary.add_row("Agent", str(agent))
    summary.add_row("Episodes", str(aggregate["num_episodes"]))
    summary.add_row("Victory Rate", f"{aggregate['victory_rate']:.1f}%")
    summary.add_row("Mean Reward", f"{aggregate['mean_reward']:.2f} ± {aggregate['std_reward']:.2f}")
    summary.add_row("Mean Steps", f"{aggregate['mean_steps']:.0f} ± {aggregate['std_steps']:.0f}")
    summary.add_row("Mean Damage", f"{aggregate['mean_damage_pct']:.1f}% ± {aggregate['std_damage_pct']:.1f}%")
    summary.add_row("Max Damage", f"{aggregate['max_damage_pct']:.1f}%")
    summary.add_row("Mean Survival", f"{aggregate['mean_survival_pct']:.1f}%")

    console.print(summary)

