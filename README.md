# LLMParticipant-Milgram

A simulation of the Milgram obedience experiment using ChatGPT-5 from OpenRouter as the participant.

## Setup

1. Install dependencies:
   ```bash
   pip install requests
   ```

2. Get an OpenRouter API key:
   - Visit https://openrouter.ai/keys
   - Create an account and generate an API key

3. Set your API key:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```
   
   Or copy `.env.example` to `.env` and add your key there.

## Usage

Run the simulation:
```bash
python main.py
```

This will run 10 trials with different experimental conditions and output results to `milgram_results.json`.

## Model Configuration

The simulation uses ChatGPT-5 via OpenRouter (`openai/gpt-5`). You can modify the model by changing the model name in the `LLMClient` initialization.

