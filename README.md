# Detective Bots Game

A Streamlit-based detective game simulation where AI agents with distinct personality types interact to identify a killer. The game showcases how language models can simulate personalities and reasoning in a structured game format.

## Overview

The Detective Bots game creates a simulated murder mystery where five AI agents (each embodying a different personality trait from the Big Five personality model) interact through a series of question-and-answer rounds. One agent is randomly selected as the killer, and the others must identify the killer through careful questioning and analysis of responses.

Each agent has a distinct personality that influences how they ask questions and respond to others:

- **Openness Agent**: Imaginative, philosophical, and intellectually curious
- **Conscientiousness Agent**: Organized, methodical, and detail-oriented
- **Extraversion Agent**: Energetic, sociable, and direct
- **Agreeableness Agent**: Cooperative, empathetic, and relationship-focused
- **Neuroticism Agent**: Vigilant, detail-sensitive, and cautious

## Key Features

- Automated game simulation using language models (GPT, Gemini, Claude)
- Agents ask questions and provide answers based on their personalities
- Voting mechanism to determine who the agents believe is the killer
- Clean, visually appealing UI with agent cards and conversation history
- Game results saved for analysis and review
- Fully automated game progression

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- API keys for language model providers (OpenAI, Google, optionally Anthropic)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shahzaib3120/detective-bots.git
   cd detective-bots
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a .env file by copying the example:

   ```bash
   cp .env.example .env
   ```

4. Edit the .env file and add your API keys:
   ```
   GOOGLE_API_KEY=your-google-api-key
   OPENAI_API_KEY=your-openai-api-key
   # Optional: ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

### Running the Game

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser, typically at http://localhost:8501

## How to Play

1. Select a language model from the dropdown (e.g., GPT, Gemini, Claude)
2. Click "Start New Game" to begin
3. The game will automatically progress through question rounds
   - One agent will ask a question each round
   - Other agents will respond based on their personalities
   - The killer will try to avoid detection while maintaining their personality
4. After 5 rounds, a voting phase will begin
   - Agents will vote on who they believe is the killer
   - If the killer receives 3 or more votes, they are caught
   - If not, the game continues for more rounds
5. Game results are saved for review

## How It Works

The game simulation uses several key components:

1. **Game Engine**: Manages the game state, agents, and core logic
2. **LLM Interface**: Handles communication with language models
3. **Prompt Templates**: Structured prompts that guide agent behavior
4. **Streamlit UI**: Interactive interface for visualization and control

The core simulation loop:

1. An agent is selected as the questioner for the current round
2. The questioner generates a question based on their personality
3. Other agents respond to the question, with the killer providing deceptive answers
4. After 5 rounds, agents vote on who they believe is the killer
5. Game concludes when the killer is correctly identified or after 10 rounds

## Game Results

Game results are saved in two formats:

- Individual JSON files for each game in the game_results directory
- A consolidated CSV file all_games.csv for statistical analysis
