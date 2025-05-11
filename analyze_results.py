import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from datetime import datetime

# Set style for visualization
plt.style.use("ggplot")
sns.set_palette("deep")
sns.set_context("notebook", font_scale=1.2)


def load_game_data():
    """Load and process game data from CSV and JSON files"""
    csv_path = "game_results/all_games.csv"

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return None, []

    # Load the CSV summary data
    summary_df = pd.read_csv(csv_path)

    # Load detailed JSON data for each game
    json_files = [f for f in os.listdir("game_results") if f.endswith(".json")]
    game_details = []

    for file in json_files:
        try:
            with open(f"game_results/{file}", "r") as f:
                game_data = json.load(f)
                game_details.append(game_data)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    print(
        f"Loaded {len(summary_df)} games from CSV and {len(game_details)} detailed game records"
    )

    return summary_df, game_details


def analyze_agent_accuracy(summary_df):
    """Analyze which agents correctly identified the killer most often"""
    agent_names = [
        col.replace("_voted_for", "")
        for col in summary_df.columns
        if col.endswith("_voted_for")
    ]
    correct_votes = {}

    for agent in agent_names:
        # Count when agent's vote matched the actual killer
        correct_votes[agent] = sum(
            summary_df[f"{agent}_voted_for"] == summary_df["actual_killer"]
        )

    # Create a DataFrame for plotting
    accuracy_df = pd.DataFrame(
        {
            "Agent": list(correct_votes.keys()),
            "Correct Votes": list(correct_votes.values()),
            "Accuracy": [
                correct / len(summary_df) * 100 for correct in correct_votes.values()
            ],
        }
    )

    # Sort by accuracy
    accuracy_df = accuracy_df.sort_values("Accuracy", ascending=False)

    # Plot the results
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Agent", y="Accuracy", data=accuracy_df)
    plt.title("Agent Accuracy in Identifying the Killer")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Agent")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("game_results/agent_accuracy.png")

    return accuracy_df


def analyze_false_accusations(summary_df, game_details):
    """Analyze which agents were falsely accused most often"""
    # Initialize vote counts for each agent
    agent_names = [
        "Openness Agent",
        "Conscientiousness Agent",
        "Extraversion Agent",
        "Agreeableness Agent",
        "Neuroticism Agent",
    ]

    false_accusations = {agent: 0 for agent in agent_names}
    total_games = len(game_details)

    # Count false accusations from game details
    for game in game_details:
        killer = game["actual_killer"]

        for agent, vote_info in game["votes"].items():
            voted_for = vote_info["vote"]
            if voted_for != killer:
                false_accusations[voted_for] = false_accusations.get(voted_for, 0) + 1

    # Create DataFrame for plotting
    false_acc_df = pd.DataFrame(
        {
            "Agent": list(false_accusations.keys()),
            "False Accusations": list(false_accusations.values()),
            "Percentage": [
                count / total_games * 100 for count in false_accusations.values()
            ],
        }
    )

    false_acc_df = false_acc_df.sort_values("False Accusations", ascending=False)

    # Plot the results
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Agent", y="False Accusations", data=false_acc_df)
    plt.title("Agents Falsely Accused as the Killer")
    plt.ylabel("Number of False Accusations")
    plt.xlabel("Agent")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("game_results/false_accusations.png")

    return false_acc_df


def analyze_model_performance(summary_df):
    """Analyze performance of different LLM models"""
    if len(summary_df) == 0:
        return pd.DataFrame()

    model_performance = (
        summary_df.groupby("model")
        .agg(
            {
                "game_id": "count",
                "correctly_identified": "mean",
                "rounds_played": "mean",
            }
        )
        .reset_index()
    )

    model_performance = model_performance.rename(
        columns={
            "game_id": "Games Played",
            "correctly_identified": "Success Rate",
            "rounds_played": "Avg Rounds",
        }
    )

    model_performance["Success Rate"] = model_performance["Success Rate"] * 100

    # Plot the results
    plt.figure(figsize=(12, 6))
    sns.barplot(x="model", y="Success Rate", data=model_performance)
    plt.title("LLM Model Performance in Solving Cases")
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("game_results/model_performance.png")

    return model_performance


def analyze_killer_detection_rate(game_details):
    """Analyze how often each agent type was caught when they were the killer"""
    agent_names = [
        "Openness Agent",
        "Conscientiousness Agent",
        "Extraversion Agent",
        "Agreeableness Agent",
        "Neuroticism Agent",
    ]

    killer_data = {
        agent: {"times_as_killer": 0, "times_caught": 0} for agent in agent_names
    }

    for game in game_details:
        killer = game["actual_killer"]
        killer_data[killer]["times_as_killer"] += 1

        if game["outcome"]["correctly_identified"]:
            killer_data[killer]["times_caught"] += 1

    # Create DataFrame
    killer_df_rows = []
    for agent, data in killer_data.items():
        if data["times_as_killer"] > 0:
            escape_rate = (1 - (data["times_caught"] / data["times_as_killer"])) * 100
            killer_df_rows.append(
                {
                    "Agent": agent,
                    "Times as Killer": data["times_as_killer"],
                    "Times Caught": data["times_caught"],
                    "Escape Rate": escape_rate,
                }
            )

    killer_df = pd.DataFrame(killer_df_rows)

    if len(killer_df) > 0:
        # Plot the results
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Agent", y="Escape Rate", data=killer_df)
        plt.title("Killer Escape Rate by Agent Type")
        plt.ylabel("Escape Rate (%)")
        plt.xlabel("Agent")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("game_results/killer_escape_rate.png")

    return killer_df


def analyze_average_rounds(summary_df):
    """Analyze the average number of rounds played before resolution"""
    avg_rounds = summary_df["rounds_played"].mean()

    # Group by correct/incorrect outcomes
    outcome_rounds = (
        summary_df.groupby("correctly_identified")["rounds_played"].mean().reset_index()
    )
    outcome_rounds["correctly_identified"] = outcome_rounds["correctly_identified"].map(
        {True: "Killer Identified", False: "Killer Escaped"}
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.barplot(x="correctly_identified", y="rounds_played", data=outcome_rounds)
    plt.title("Average Rounds Played by Game Outcome")
    plt.ylabel("Average Rounds")
    plt.xlabel("Outcome")
    plt.tight_layout()
    plt.savefig("game_results/average_rounds.png")

    return avg_rounds, outcome_rounds


def analyze_conversation_patterns(game_details):
    """Analyze conversation patterns for killers versus non-killers"""
    killer_answer_lengths = []
    non_killer_answer_lengths = []

    for game in game_details:
        killer = game["actual_killer"]

        for round_data in game["rounds"]:
            for responder, response in round_data["answers"].items():
                if responder == killer:
                    killer_answer_lengths.append(len(response))
                else:
                    non_killer_answer_lengths.append(len(response))

    # Calculate average lengths
    avg_killer_length = np.mean(killer_answer_lengths) if killer_answer_lengths else 0
    avg_non_killer_length = (
        np.mean(non_killer_answer_lengths) if non_killer_answer_lengths else 0
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    data = pd.DataFrame(
        {
            "Role": ["Killer", "Non-Killer"],
            "Average Answer Length": [avg_killer_length, avg_non_killer_length],
        }
    )
    sns.barplot(x="Role", y="Average Answer Length", data=data)
    plt.title("Average Answer Length: Killers vs. Non-Killers")
    plt.ylabel("Average Character Count")
    plt.tight_layout()
    plt.savefig("game_results/answer_length_comparison.png")

    return avg_killer_length, avg_non_killer_length


def generate_report(summary_df, game_details):
    """Generate a comprehensive analysis report"""
    report_path = "game_results/analysis_report.html"

    # Get all analysis results
    agent_accuracy = analyze_agent_accuracy(summary_df)
    false_accusations = analyze_false_accusations(summary_df, game_details)
    model_performance = analyze_model_performance(summary_df)
    killer_detection = analyze_killer_detection_rate(game_details)
    avg_rounds, outcome_rounds = analyze_average_rounds(summary_df)
    avg_killer_length, avg_non_killer_length = analyze_conversation_patterns(
        game_details
    )

    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detective Bots Game Analysis</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #333366; }}
            .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 15px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .highlight {{ background-color: #ffffcc; }}
            .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Detective Bots Game Analysis Report</h1>
            <p>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="section summary">
                <h2>Executive Summary</h2>
                <p>Analysis based on {len(summary_df)} games:</p>
                <ul>
                    <li>Overall killer identification success rate: <strong>{summary_df["correctly_identified"].mean() * 100:.1f}%</strong></li>
                    <li>Average number of rounds per game: <strong>{avg_rounds:.1f}</strong></li>
                    <li>Best performing agent: <strong>{agent_accuracy.iloc[0]["Agent"]}</strong> with accuracy of <strong>{agent_accuracy.iloc[0]["Accuracy"]:.1f}%</strong></li>
                    <li>Most falsely accused agent: <strong>{false_accusations.iloc[0]["Agent"]}</strong> with <strong>{false_accusations.iloc[0]["False Accusations"]}</strong> false accusations</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Agent Accuracy Analysis</h2>
                <p>This analysis shows which agents were most successful at correctly identifying the killer.</p>
                <img src="agent_accuracy.png" alt="Agent Accuracy Chart">
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Correct Votes</th>
                        <th>Accuracy (%)</th>
                    </tr>
    """

    # Add agent accuracy table rows
    for _, row in agent_accuracy.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row["Agent"]}</td>
                        <td>{row["Correct Votes"]}</td>
                        <td>{row["Accuracy"]:.1f}%</td>
                    </tr>
        """

    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>False Accusations Analysis</h2>
                <p>This analysis shows which agents were most often falsely accused of being the killer.</p>
                <img src="false_accusations.png" alt="False Accusations Chart">
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>False Accusations</th>
                        <th>Percentage</th>
                    </tr>
    """

    # Add false accusations table rows
    for _, row in false_accusations.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row["Agent"]}</td>
                        <td>{row["False Accusations"]}</td>
                        <td>{row["Percentage"]:.1f}%</td>
                    </tr>
        """

    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>LLM Model Performance</h2>
                <p>This analysis compares the performance of different language models used in the game.</p>
                <img src="model_performance.png" alt="Model Performance Chart">
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Games Played</th>
                        <th>Success Rate (%)</th>
                        <th>Avg Rounds</th>
                    </tr>
    """

    # Add model performance table rows
    for _, row in model_performance.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row["model"]}</td>
                        <td>{row["Games Played"]}</td>
                        <td>{row["Success Rate"]:.1f}%</td>
                        <td>{row["Avg Rounds"]:.1f}</td>
                    </tr>
        """

    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Killer Detection Rate</h2>
                <p>This analysis shows how often each agent type was caught when they were the killer.</p>
                <img src="killer_escape_rate.png" alt="Killer Escape Rate Chart">
                <table>
                    <tr>
                        <th>Agent</th>
                        <th>Times as Killer</th>
                        <th>Times Caught</th>
                        <th>Escape Rate (%)</th>
                    </tr>
    """

    # Add killer detection table rows
    for _, row in killer_detection.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row["Agent"]}</td>
                        <td>{row["Times as Killer"]}</td>
                        <td>{row["Times Caught"]}</td>
                        <td>{row["Escape Rate"]:.1f}%</td>
                    </tr>
        """

    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Game Length Analysis</h2>
                <p>This analysis shows the average number of rounds played before the game ended.</p>
                <img src="average_rounds.png" alt="Average Rounds Chart">
                <table>
                    <tr>
                        <th>Outcome</th>
                        <th>Average Rounds</th>
                    </tr>
    """

    # Add rounds table rows
    for _, row in outcome_rounds.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row["correctly_identified"]}</td>
                        <td>{row["rounds_played"]:.1f}</td>
                    </tr>
        """

    html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>Conversation Pattern Analysis</h2>
                <p>This analysis compares the communication patterns between killers and non-killers.</p>
                <img src="answer_length_comparison.png" alt="Answer Length Comparison Chart">
                <p>Average answer length for killers: <strong>{avg_killer_length:.1f} characters</strong></p>
                <p>Average answer length for non-killers: <strong>{avg_non_killer_length:.1f} characters</strong></p>
                <p>Difference: <strong>{avg_killer_length - avg_non_killer_length:.1f} characters</strong></p>
            </div>
            
            <div class="section">
                <h2>Conclusion</h2>
                <p>Based on the analysis of {len(summary_df)} games, we can draw the following conclusions:</p>
                <ul>
                    <li>The killer was successfully identified in {summary_df["correctly_identified"].sum()} out of {len(summary_df)} games ({summary_df["correctly_identified"].mean() * 100:.1f}%).</li>
                    <li><strong>{agent_accuracy.iloc[0]["Agent"]}</strong> was the most successful at identifying the killer, while <strong>{agent_accuracy.iloc[-1]["Agent"]}</strong> was the least successful.</li>
                    <li><strong>{false_accusations.iloc[0]["Agent"]}</strong> was most often falsely accused of being the killer.</li>
                    <li>When <strong>{killer_detection.iloc[0]["Agent"] if len(killer_detection) > 0 else "N/A"}</strong> was the killer, they had the highest chance of escaping detection.</li>
                    <li>Killers' responses were {avg_killer_length - avg_non_killer_length:.1f} characters {"longer" if avg_killer_length > avg_non_killer_length else "shorter"} than non-killers on average.</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    # Write the HTML report
    with open(report_path, "w") as f:
        f.write(html_content)

    print(f"Analysis report generated at {report_path}")


def main():
    """Main function to perform all analyses"""
    print("Loading game data...")
    summary_df, game_details = load_game_data()

    if summary_df is None or len(summary_df) == 0:
        print("No game data found. Please run some games first.")
        return

    print(f"Analyzing {len(summary_df)} games...")

    # Create output directory if it doesn't exist
    os.makedirs("game_results", exist_ok=True)

    # Generate the comprehensive report
    generate_report(summary_df, game_details)

    print("Analysis complete! Report and visualizations saved to game_results/ folder.")


if __name__ == "__main__":
    main()
