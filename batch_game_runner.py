import os
import json
import uuid
import time
import random
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
import itertools

from game_engine import GameEngine
from llm_interface import get_available_models

# Load environment variables
load_dotenv()


def run_single_game(model_name, game_id=None):
    """Run a single game with the specified model and return the results"""
    if game_id is None:
        game_id = str(uuid.uuid4())

    print(f"Starting game {game_id} with model {model_name}")

    # Initialize game engine
    game_engine = GameEngine(model_name)
    game_log = []

    # Run 5 rounds
    for round_num in range(1, 6):
        try:
            round_results = game_engine.run_round(round_num)
            game_log.append(round_results)
        except Exception as e:
            print(f"Error in round {round_num} of game {game_id}: {str(e)}")
            return None

    # Conduct voting
    try:
        votes = game_engine.conduct_voting()

        # Count votes
        vote_counts = {}
        for agent, vote_info in votes.items():
            target = vote_info["vote"]
            vote_counts[target] = vote_counts.get(target, 0) + 1

        # Check if killer has been identified
        killer = game_engine.killer.name
        killer_votes = vote_counts.get(killer, 0)
        correctly_identified = killer_votes >= 3

        # Create outcome
        outcome = {
            "correctly_identified": correctly_identified,
            "vote_distribution": vote_counts,
            "rounds_played": 5,
        }

        # If not identified, run 5 more rounds
        if not correctly_identified:
            for round_num in range(6, 11):
                try:
                    round_results = game_engine.run_round(round_num)
                    game_log.append(round_results)
                except Exception as e:
                    print(f"Error in round {round_num} of game {game_id}: {str(e)}")
                    return None

            # Conduct final voting
            try:
                votes = game_engine.conduct_voting()

                # Count votes
                vote_counts = {}
                for agent, vote_info in votes.items():
                    target = vote_info["vote"]
                    vote_counts[target] = vote_counts.get(target, 0) + 1

                # Check if killer has been identified
                killer_votes = vote_counts.get(killer, 0)
                correctly_identified = killer_votes >= 3

                # Update outcome
                outcome = {
                    "correctly_identified": correctly_identified,
                    "vote_distribution": vote_counts,
                    "rounds_played": 10,
                }
            except Exception as e:
                print(f"Error in final voting of game {game_id}: {str(e)}")
                return None

        # Compile results
        results = {
            "game_id": game_id,
            "model": model_name,
            "actual_killer": killer,
            "rounds": game_log,
            "votes": votes,
            "outcome": outcome,
        }

        # Save results
        save_game_results(results)

        print(
            f"Game {game_id} completed. Killer: {killer}. Correctly identified: {correctly_identified} by {model_name} LLM."
        )
        return results
    except Exception as e:
        print(f"Error in game {game_id}: {str(e)}")
        return None


def save_game_results(results):
    """Save the game results to a JSON file and update the CSV summary"""
    # Create directory if it doesn't exist
    os.makedirs("game_results", exist_ok=True)

    # Save to JSON
    filename = f"game_results/game_{results['game_id']}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    # Create row for results
    data = {
        "game_id": results["game_id"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": results["model"],
        "actual_killer": results["actual_killer"],
        "rounds_played": results["outcome"]["rounds_played"],
        "correctly_identified": results["outcome"]["correctly_identified"],
    }

    # Add vote information
    for agent, vote_info in results["votes"].items():
        data[f"{agent}_voted_for"] = vote_info["vote"]

    # Check if file exists
    csv_file = "game_results/all_games.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])

    df.to_csv(csv_file, index=False)


def get_existing_game_counts():
    """Get the count of existing games for each model from the CSV"""
    csv_file = "game_results/all_games.csv"
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            return df.groupby("model").size().to_dict()
        except Exception as e:
            print(f"Error reading existing games CSV: {e}")
            return {}
    else:
        return {}


def run_batch_games(models=None, games_per_model=25, parallel=False, max_workers=4):
    """Run a batch of games for each specified model"""
    if models is None:
        models = get_available_models()

    # Get existing game counts by model
    existing_counts = get_existing_game_counts()

    # Calculate how many games to run for each model
    games_to_run = {}
    for model in models:
        existing = existing_counts.get(model, 0)
        remaining = max(0, min(games_per_model, 25 - existing))
        games_to_run[model] = remaining

    # Filter out models that already have 25 or more games
    active_models = [model for model in models if games_to_run[model] > 0]

    if not active_models:
        print(
            "All specified models already have 25 or more games. No new games will be run."
        )
        return []

    total_games = sum(games_to_run.values())

    print(f"Existing games detected: {existing_counts}")
    print(f"New games to run: {games_to_run}")
    print(
        f"Preparing to run {total_games} new games across {len(active_models)} models"
    )

    results = []

    if parallel:
        # Create a list of all game jobs in round-robin fashion with adjusted counts
        all_jobs = []

        # Create a more balanced distribution of jobs
        # First create a list of (model, count) tuples
        model_counts = [(model, games_to_run[model]) for model in active_models]

        # Create jobs using round-robin allocation across models
        while any(count > 0 for _, count in model_counts):
            for i, (model, count) in enumerate(model_counts):
                if count > 0:
                    game_id = str(uuid.uuid4())
                    all_jobs.append((model, game_id))
                    model_counts[i] = (model, count - 1)

        # Run games in parallel with improved allocation strategy
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs in order - this ensures we're not hitting the same API with multiple workers
            futures = []
            batch_size = min(max_workers, len(active_models))

            # Submit jobs in batches to ensure model diversity
            for i in range(0, len(all_jobs), batch_size):
                batch = all_jobs[i : i + batch_size]
                for model, game_id in batch:
                    futures.append(executor.submit(run_single_game, model, game_id))

                # Wait for this batch to complete before starting the next
                # to ensure we don't have too many concurrent requests to the same API
                if i + batch_size < len(all_jobs):
                    time.sleep(2)  # Short delay between batches

            # Process results as they complete
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Running games"
            ):
                result = future.result()
                if result:
                    results.append(result)
    else:
        # Run games sequentially with round-robin through models
        # This distributes API calls across models more evenly

        # Create a cycle with models that need more games
        model_cycle = itertools.cycle(active_models)
        completed_counts = {model: 0 for model in active_models}

        with tqdm(total=total_games, desc="Running games") as pbar:
            while sum(completed_counts.values()) < total_games:
                # Get next model in round-robin fashion
                model = next(model_cycle)

                # Skip if we've completed all games for this model
                if completed_counts[model] >= games_to_run[model]:
                    continue

                # Run a game with the selected model
                result = run_single_game(model)
                if result:
                    results.append(result)
                    completed_counts[model] += 1
                    pbar.update(1)

                # Add a small delay between games to avoid rate limiting
                time.sleep(1)

    print(f"Batch run completed. {len(results)} games successfully processed.")

    # Analyze success rate by model for newly run games
    if results:
        df = pd.DataFrame(
            [
                {
                    "model": r["model"],
                    "correctly_identified": r["outcome"]["correctly_identified"],
                }
                for r in results
            ]
        )

        success_by_model = (
            df.groupby("model")["correctly_identified"]
            .agg(["count", "mean"])
            .reset_index()
        )
        success_by_model["success_rate"] = success_by_model["mean"] * 100
        success_by_model = success_by_model.rename(
            columns={"count": "games", "mean": "success_rate_decimal"}
        )
        success_by_model = success_by_model[["model", "games", "success_rate"]]

        print("\nSuccess rate by model (newly run games):")
        print(success_by_model)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multiple detective games automatically"
    )
    parser.add_argument("--models", nargs="+", help="List of models to use")
    parser.add_argument(
        "--games", type=int, default=25, help="Number of games per model"
    )
    parser.add_argument("--parallel", action="store_true", help="Run games in parallel")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes if running in parallel",
    )

    args = parser.parse_args()

    if args.models:
        available_models = get_available_models()
        for model in args.models:
            if model not in available_models:
                print(
                    f"Warning: Model {model} is not in the available models list: {available_models}"
                )
        run_batch_games(args.models, args.games, args.parallel, args.workers)
    else:
        run_batch_games(
            games_per_model=args.games, parallel=args.parallel, max_workers=args.workers
        )
