import os
import json
from datetime import datetime
import pandas as pd


def save_game_results(results, csv_file_path="game_results/all_games.csv"):
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
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])

    df.to_csv(csv_file_path, index=False)
