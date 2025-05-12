import streamlit as st
import json
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from game_engine import GameEngine
from llm_interface import get_available_models
from utils import save_game_results

# Load environment variables
load_dotenv()


def apply_shared_styles():
    """Apply shared CSS styles for cards and UI elements"""
    st.markdown(
        """
    <style>
    /* Shared card styles */
    .styled-card {
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        display: flex;
        flex-direction: column;
    }
    
    /* Agent card specific styles */
    .agent-card {
        height: 120px;
    }
    
    /* Text styling classes */
    .card-title {
        font-size: 16px;
        font-weight: 600;
        margin: 0;
        padding: 0;
    }
    .card-subtitle {
        font-size: 13px;
        font-style: italic;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .card-status {
        font-size: 14px;
        font-weight: 500;
        margin-top: auto;
        padding-top: 5px;
    }
    .highlighted-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 5px;
    }
    
    /* Vote card specific styles */
    .vote-card {
        text-align: center;
        transition: transform 0.2s ease;
    }
    .vote-card:hover {
        transform: translateY(-2px);
    }
    .vote-count {
        font-size: 24px;
        font-weight: 700;
        margin: 8px 0;
    }
    .detailed-vote {
        padding: 12px;
    }
    .vote-reasoning {
        font-size: 14px;
        font-style: italic;
        margin-top: 8px;
        padding: 8px;
        border-radius: 6px;
        background-color: rgba(0,0,0,0.03);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Page configuration
st.set_page_config(page_title="Detective Game Simulation", layout="wide")
apply_shared_styles()


# Define agent colors
AGENT_COLORS = {
    "Openness Agent": "#3366CC",  # Blue
    "Conscientiousness Agent": "#DC3912",  # Red
    "Extraversion Agent": "#FF9900",  # Orange
    "Agreeableness Agent": "#109618",  # Green
    "Neuroticism Agent": "#990099",  # Purple
}

# Inside the session state
if "game_initialized" not in st.session_state:
    st.session_state.game_initialized = False
    st.session_state.game_id = None
    st.session_state.current_round = 0
    st.session_state.game_complete = False
    st.session_state.game_log = []
    st.session_state.votes = {}
    st.session_state.game_outcome = None
    st.session_state.current_questioner = None
    st.session_state.round_in_progress = False
    st.session_state.auto_advance_time = None
    st.session_state.next_action = "round"


def initialize_new_game(model_name):
    """Initialize a new game session"""
    st.session_state.game_id = str(uuid.uuid4())
    st.session_state.model_name = model_name
    st.session_state.game_engine = GameEngine(model_name)
    st.session_state.game_initialized = True
    st.session_state.current_round = 1
    st.session_state.game_complete = False
    st.session_state.game_log = []
    st.session_state.votes = {}
    st.session_state.game_outcome = None
    st.session_state.round_in_progress = False
    st.session_state.current_questioner = None
    st.session_state.voting_conducted = False
    st.session_state.next_action = "round"

    # No delay or timer needed - start immediately
    st.rerun()  # Trigger immediate rerun to start first round


def run_game_round():
    """Run a single round of the game"""
    if st.session_state.round_in_progress:
        return

    # Set the current questioner at the start of the round
    game_engine = st.session_state.game_engine
    questioner_idx = (st.session_state.current_round - 1) % len(game_engine.agents)
    st.session_state.current_questioner = game_engine.agents[questioner_idx].name

    st.session_state.round_in_progress = True

    # Create a placeholder for progress feedback
    progress_placeholder = st.empty()
    progress_placeholder.info(f"Running round {st.session_state.current_round}...")

    try:
        # Use a separate thread for LLM processing to keep UI responsive
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                game_engine.run_round, st.session_state.current_round
            )
            round_results = future.result()

        st.session_state.game_log.append(round_results)
        st.session_state.current_round += 1

        # After round 5, conduct voting after each round
        if st.session_state.current_round > 5:
            st.session_state.next_action = "voting"
        else:
            st.session_state.next_action = "round"

        # Clear the progress message
        progress_placeholder.empty()

    except Exception as e:
        st.error(f"Error in game round: {str(e)}")
    finally:
        st.session_state.round_in_progress = False
        st.rerun()


def conduct_voting():
    """Conduct the voting phase after round 5 and after each subsequent round, stopping if a majority is found."""
    if st.session_state.round_in_progress:
        return

    st.session_state.round_in_progress = True
    game_engine = st.session_state.game_engine

    # Create a placeholder for progress feedback
    progress_placeholder = st.empty()
    progress_placeholder.info("Conducting voting phase...")

    try:
        # Use a separate thread for LLM processing to keep UI responsive
        with ThreadPoolExecutor() as executor:
            future = executor.submit(game_engine.conduct_voting)
            votes = future.result()

        st.session_state.votes = votes
        st.session_state.voting_conducted = True

        # Count votes
        vote_counts = {}
        for agent, vote_info in st.session_state.votes.items():
            target = vote_info["vote"]
            vote_counts[target] = vote_counts.get(target, 0) + 1
        st.session_state.vote_counts = vote_counts

        # Check for majority (3+ votes for any agent)
        majority_agent = None
        for agent, count in vote_counts.items():
            if count >= 3:
                majority_agent = agent
                break

        # Always store vote information regardless of outcome
        st.session_state.game_outcome = {
            "majority_found": majority_agent is not None,
            "majority_agent": majority_agent,
            "vote_distribution": vote_counts,
            "rounds_played": st.session_state.current_round - 1,
            "correctly_identified": majority_agent == game_engine.killer.name
            if majority_agent
            else False,
        }

        # Save after each voting
        results = {
            "game_id": st.session_state.game_id,
            "model": st.session_state.model_name,
            "actual_killer": st.session_state.game_engine.killer.name,
            "rounds": st.session_state.game_log,
            "votes": st.session_state.votes,
            "outcome": st.session_state.game_outcome,
        }
        save_game_results(results)

        if majority_agent or st.session_state.current_round > 20:
            st.session_state.game_complete = True
        else:
            # Continue with next round
            st.session_state.next_action = "round"

        # Clear the progress message
        progress_placeholder.empty()

    except Exception as e:
        st.error(f"Error in voting phase: {str(e)}")
    finally:
        st.session_state.round_in_progress = False
        st.rerun()


# UI Layout
st.title("Detective Game Simulation")

# Model selection
available_models = get_available_models()
selected_model = st.selectbox("Select Language Model", available_models)

# Game controls
col1, col2 = st.columns([1, 3])
with col1:
    if not st.session_state.game_initialized:
        if st.button("Start New Game"):
            initialize_new_game(selected_model)
    else:
        # Manual controls for override
        if not st.session_state.game_complete:
            st.write("Game automatically progressing...")
            if st.button("Skip to Next Round"):
                if st.session_state.current_round > 5:
                    conduct_voting()
                else:
                    run_game_round()
        else:
            if st.button("Start New Game"):
                initialize_new_game(selected_model)

# Game state display
if st.session_state.game_initialized:
    with col2:
        st.write(f"Game ID: {st.session_state.game_id}")
        st.write(
            f"Current Round: {st.session_state.current_round - 1 if st.session_state.current_round > 0 else 0}"
        )
        st.write(f"Model: {st.session_state.model_name}")

        if st.session_state.game_complete:
            killer = st.session_state.game_engine.killer.name
            if st.session_state.game_outcome["correctly_identified"]:
                st.success(f"The killer ({killer}) was successfully identified!")
            else:
                st.error(f"The killer ({killer}) was not identified.")

            st.write("Vote Distribution:")
            for agent, count in st.session_state.game_outcome[
                "vote_distribution"
            ].items():
                st.write(f"{agent}: {count} votes")

    # Display agent information with boxes and highlighting
    if hasattr(st.session_state, "game_engine"):
        st.subheader("Agents")

        # Create a row of columns for the agents
        cols = st.columns(5)

        for i, agent in enumerate(st.session_state.game_engine.agents):
            with cols[i]:
                agent_color = AGENT_COLORS[agent.name]
                is_questioner = agent.name == st.session_state.current_questioner
                is_killer = (
                    agent == st.session_state.game_engine.killer
                    and st.session_state.game_complete
                )

                # Determine background color - subtle gradient for visual appeal
                bg_color = (
                    f"linear-gradient(to bottom, {agent_color}15, {agent_color}30)"
                )
                if is_questioner:
                    bg_color = (
                        f"linear-gradient(to bottom, {agent_color}30, {agent_color}45)"
                    )

                # Build status text
                status_text = ""
                if (
                    st.session_state.next_action == "voting"
                    and not st.session_state.game_complete
                ):
                    status_text = "Currently voting"
                elif is_questioner:
                    status_text = "Currently asking"
                if is_killer:
                    status_text += (
                        " <span style='color: gold; font-weight: bold;'>🗡️ KILLER</span>"
                    )

                # Render the card with clean HTML using our shared classes
                st.markdown(
                    f"""
                    <div class="styled-card agent-card" style="border-left: 4px solid {agent_color}; background: {bg_color};">
                        <p class="card-title" style="color: {agent_color};">{agent.name}</p>
                        <p class="card-subtitle" style="color: {agent_color}80;">{agent.personality_type}</p>
                        <p class="card-status" style="color: {agent_color};">{status_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Display game log with colored text
    st.subheader("Game Log")
    game_log_container = st.container()

    with game_log_container:
        for round_idx, round_data in enumerate(st.session_state.game_log):
            questioner = round_data["questioner"]
            questioner_color = AGENT_COLORS.get(questioner, "#000000")

            st.markdown(f"### Round {round_idx + 1}")
            st.markdown(
                f"<div style='background-color: rgba({','.join(str(int(questioner_color.lstrip('#')[i : i + 2], 16)) for i in (0, 2, 4))},0.1); padding: 10px; border-radius: 5px; border-left: 5px solid {questioner_color};'>"
                f"<span style='color: {questioner_color}; font-weight: bold;'>{questioner}</span> asks: \"{round_data['question']}\""
                f"</div>",
                unsafe_allow_html=True,
            )

            for responder, response in round_data["answers"].items():
                responder_color = AGENT_COLORS.get(responder, "#000000")
                st.markdown(
                    f"<div style='margin-left: 20px; margin-top: 10px; background-color: rgba({','.join(str(int(responder_color.lstrip('#')[i : i + 2], 16)) for i in (0, 2, 4))},0.05); padding: 10px; border-radius: 5px; border-left: 3px solid {responder_color};'>"
                    f"<span style='color: {responder_color}; font-weight: bold;'>{responder}</span>: {response}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

        # Display voting results if game complete
        if hasattr(st.session_state, "votes") and st.session_state.votes:
            st.markdown("### Voting Results")

            # Define additional styles for vote cards
            st.markdown(
                """
            <style>
            .vote-card {
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.15);
                padding: 15px;
                margin-bottom: 15px;
                text-align: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                transition: transform 0.2s ease;
            }
            .vote-card:hover {
                transform: translateY(-2px);
            }
            .vote-count {
                font-size: 24px;
                font-weight: 700;
                margin: 8px 0;
            }
            .vote-badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                margin-top: 5px;
            }
            .detailed-vote {
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 10px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .vote-reasoning {
                font-size: 14px;
                font-style: italic;
                margin-top: 8px;
                padding: 8px;
                border-radius: 6px;
                background-color: rgba(0,0,0,0.03);
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # First show vote summary
            st.markdown("#### Vote Summary")
            vote_cols = st.columns(len(st.session_state.game_engine.agents))

            for i, agent in enumerate(st.session_state.game_engine.agents):
                with vote_cols[i]:
                    agent_color = AGENT_COLORS[agent.name]
                    votes_received = st.session_state.vote_counts.get(agent.name, 0)
                    is_killer = agent == st.session_state.game_engine.killer

                    # Set background color based on vote count and killer status
                    if is_killer:
                        bg_color = "linear-gradient(to bottom, rgba(255,215,0,0.15), rgba(255,215,0,0.3))"
                        border_style = f"border-left: 4px solid gold; border-top: 1px solid {agent_color}40"
                    else:
                        bg_color = f"linear-gradient(to bottom, {agent_color}10, {agent_color}25)"
                        border_style = f"border-left: 4px solid {agent_color}"

                    # Create a badge for majority vote or killer
                    badges = []
                    if votes_received >= 3:
                        badges.append(
                            '<span class="highlighted-badge" style="background-color: #4CAF50; color: white;">Majority Vote</span>'
                        )
                    if is_killer:
                        badges.append(
                            '<span class="highlighted-badge" style="background-color: gold; color: #333;">🗡️ Killer</span>'
                        )

                    badge_html = "".join(badges)

                    st.markdown(
                        f"""
                        <div class="styled-card vote-card" style="{border_style}; background: {bg_color};">
                            <p class="card-title" style="color: {agent_color};">{agent.name}</p>
                            <div class="vote-count" style="color: {agent_color};">{votes_received}</div>
                            <p style="color: {agent_color}80;">votes received</p>
                            {badge_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Then show detailed voting with matching style
            st.markdown("#### Detailed Votes")
            for agent, vote_info in st.session_state.votes.items():
                voter_color = AGENT_COLORS.get(agent, "#000000")
                voted_for = vote_info["vote"]
                voted_for_color = AGENT_COLORS.get(voted_for, "#000000")

                # Check if voted for the actual killer
                voted_for_killer = voted_for == st.session_state.game_engine.killer.name
                target_indicator = "🗡️" if voted_for_killer else ""

                # Create a gradient background based on voter's color
                bg_color = (
                    f"linear-gradient(to right, {voter_color}15, {voter_color}05)"
                )

                st.markdown(
                    f"""
                    <div class="styled-card detailed-vote" style="border-left: 4px solid {voter_color}; background: {bg_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: {voter_color}; font-weight: 600;">{agent}</span>
                            <span style="color: #666; font-size: 14px;">voted for</span>
                            <span style="color: {voted_for_color}; font-weight: 600;">{voted_for} {target_indicator}</span>
                        </div>
                        <div class="vote-reasoning">{vote_info["reasoning"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# Auto-advance logic
if st.session_state.game_initialized and not st.session_state.game_complete:
    # First, update questioner state before doing anything else
    if not st.session_state.round_in_progress and not hasattr(
        st.session_state, "questioner_updated"
    ):
        # Set the current questioner before running the round
        if st.session_state.next_action == "round":
            game_engine = st.session_state.game_engine
            questioner_idx = (st.session_state.current_round - 1) % len(
                game_engine.agents
            )
            st.session_state.current_questioner = game_engine.agents[
                questioner_idx
            ].name
            st.session_state.questioner_updated = True
            st.rerun()  # Rerun to show the updated questioner UI

    # Then process the next action
    elif not st.session_state.round_in_progress:
        # Reset the questioner update flag
        st.session_state.questioner_updated = False

        # Execute the next action
        if st.session_state.next_action == "voting":
            conduct_voting()
        else:
            run_game_round()

# Force frequent UI updates during game
if st.session_state.game_initialized and not st.session_state.game_complete:
    # Use a progress indicator instead of sleep
    with st.empty():
        if st.session_state.round_in_progress:
            st.info(f"Round {st.session_state.current_round} in progress...")
        else:
            st.write(f"Waiting for next action... ({st.session_state.next_action})")
