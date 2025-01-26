# Discord LLM Bot

A Discord bot that integrates with multiple LLM providers to answer user questions when mentioned.

## Features

- Supports multiple LLM providers:
  - OpenAI (and OpenAI-compatible APIs)
  - Ollama (local LLMs)
  - Google Gemini
  - Deepseek
- Automatic response chunking for long messages
- Configurable default provider
- Error handling and logging
- Selective message processing (only responds to direct mentions and commands)

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.sample` to `.env` and configure your environment variables
4. Run the bot:
   ```bash
   python src/bot.py
   ```

## Configuration

### Environment Variables

| Variable               | Description                                                                 | Required | Default Value                     |
|------------------------|-----------------------------------------------------------------------------|----------|-----------------------------------|
| `DISCORD_TOKEN`        | Your Discord bot token                                                      | Yes      |                                   |
| `DEFAULT_LLM_PROVIDER` | Default LLM provider to use (`ollama`, `openai`, `gemini`, or `deepseek`)   | Yes      |                                   |
| `OPENAI_API_KEY`       | OpenAI API key                                                              | If using OpenAI |                               |
| `OPENAI_API_BASE`      | OpenAI API base URL (for OpenAI-compatible APIs)                            | No       | `https://api.openai.com/v1`       |
| `OPENAI_MODEL`         | OpenAI model to use                                                         | No       | `gpt-3.5-turbo`                   |
| `OLLAMA_API_URL`       | Ollama API URL                                                              | If using Ollama | `http://localhost:11434`      |
| `OLLAMA_MODEL`         | Ollama model to use                                                         | No       | `deepseek-r1:14b`                 |
| `GEMINI_API_KEY`       | Google Gemini API key                                                       | If using Gemini |                               |
| `GEMINI_MODEL`         | Gemini model to use                                                         | No       | `gemini-1.5-flash`                |
| `DEEPSEEK_API_KEY`     | Deepseek API key                                                            | If using Deepseek |                             |

## Usage

1. Invite the bot to your Discord server
2. Mention the bot in a message followed by your question
3. The bot will only respond to direct mentions and commands
4. The bot will respond with an answer from the configured LLM provider
5. The bot is context aware and will respond follow up questions in the conversation

Example:
```
@LLMBot What is the capital of France?
```

## Troubleshooting

- Ensure all required environment variables are set
- Check the logs for any error messages
- Verify your API keys are valid
- Make sure the bot has proper permissions in your Discord server

## Contributing

Contributions are welcome! Please open an issue or pull request for any improvements or bug fixes.

## License

[MIT](LICENSE)
