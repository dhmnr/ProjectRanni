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
) -> Union[RandomAgent, BaseAgent, TemporalAgent]:
    """Create agent based on config.

    Args:
        config: Evaluation config dict
        model: Optional loaded model (not needed for random agent)

    Returns:
        Agent instance
    """
    agent_type = config.get("agent", {}).get("type", "model")
    action_threshold = config.get("evaluation", {}).get("action_threshold", 0.5)

    # Get frame shape from config or use default
    frame_shape = tuple(
        config.get("evaluation", {}).get("frame_shape", [3, 144, 256])
    )

    if agent_type == "random":
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
            )
        else:
            return BaseAgent(
                model=model,
                action_threshold=action_threshold,
                frame_shape=frame_shape,
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
        from eldengym import EldenGymEnv
    except ImportError:
        raise ImportError(
            "eldengym not installed. Install with: pip install eldengym"
        )

    env_config = config.get("environment", {})

    # Required parameters
    scenario_name = env_config.get("scenario_name", "margit")
    siphon_config = env_config.get("siphon_config")
    keybinds = env_config.get("keybinds")

    if not siphon_config:
        raise ValueError("environment.siphon_config is required")
    if not keybinds:
        raise ValueError("environment.keybinds is required")

    # Optional parameters
    host = env_config.get("host", "localhost:50051")
    max_steps = env_config.get("max_steps", None)
    memory_attributes = env_config.get("memory_attributes", None)

    console.print(f"[cyan]Creating EldenGym environment...[/cyan]")
    console.print(f"  Scenario: {scenario_name}")
    console.print(f"  Host: {host}")
    console.print(f"  Siphon config: {siphon_config}")
    console.print(f"  Keybinds: {keybinds}")

    env = EldenGymEnv(
        scenario_name=scenario_name,
        keybinds_filepath=keybinds,
        siphon_config_filepath=siphon_config,
        memory_attributes=memory_attributes,
        host=host,
        max_steps=max_steps,
    )

    console.print("[green]✓ Environment created[/green]")
    return env


def run_episode(
    env,
    agent: Union[RandomAgent, BaseAgent, TemporalAgent],
    render: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single evaluation episode.

    Args:
        env: EldenGym environment
        agent: Agent to evaluate
        render: Whether to render frames
        max_steps: Maximum steps per episode (overrides env setting)

    Returns:
        Episode metrics dict
    """
    # Reset environment and agent
    obs, info = env.reset()
    agent.reset()

    episode_reward = 0.0
    episode_steps = 0
    start_time = time.time()

    # Track metrics
    initial_player_hp = info.get("player_hp", 0)
    initial_target_hp = info.get("target_hp", 0)
    min_target_hp = initial_target_hp

    done = False
    while not done:
        # Get action from agent
        # Combine observation with info for state-based agents
        obs_dict = {"frame": obs, **info} if isinstance(obs, np.ndarray) else {**obs, **info}
        action = agent.act(obs_dict)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        episode_steps += 1

        # Track target HP
        target_hp = info.get("target_hp", min_target_hp)
        min_target_hp = min(min_target_hp, target_hp)

        # Render if requested
        if render:
            env.render()

        # Check max steps
        if max_steps and episode_steps >= max_steps:
            break

    episode_time = time.time() - start_time

    # Calculate metrics
    player_hp_remaining = info.get("player_hp", 0)
    target_hp_remaining = info.get("target_hp", 0)

    # Damage dealt to target (as percentage)
    if initial_target_hp > 0:
        damage_dealt_pct = (initial_target_hp - min_target_hp) / initial_target_hp * 100
    else:
        damage_dealt_pct = 0.0

    # Survival (player HP remaining as percentage)
    if initial_player_hp > 0:
        survival_pct = player_hp_remaining / initial_player_hp * 100
    else:
        survival_pct = 0.0

    return {
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "episode_time": episode_time,
        "damage_dealt_pct": damage_dealt_pct,
        "survival_pct": survival_pct,
        "target_hp_remaining": target_hp_remaining,
        "player_hp_remaining": player_hp_remaining,
        "terminated": terminated,
        "truncated": truncated,
        "victory": target_hp_remaining <= 0,
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

    # Create agent
    agent = create_agent(config, model)
    console.print(f"[green]✓ Agent created: {agent}[/green]")

    # Create environment
    env = create_environment(config)

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

            try:
                metrics = run_episode(
                    env=env,
                    agent=agent,
                    render=render,
                    max_steps=max_steps,
                )
                metrics["episode"] = ep_idx + 1
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

