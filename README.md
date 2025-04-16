# Novel Translation Pipeline

This project provides an automated pipeline for translating Korean novel chapters to English using OpenAI's GPT-4 model.

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

1. Place your Korean novel chapters in the `chapters/` directory.

2. Run the translation script:

```bash
python translate_novel.py
```

3. Translated chapters will be saved in the `chapters_translated/` directory with "\_EN" suffix.

## Features

- Maintains translation consistency using context from previous chapters
- Processes chapters in numerical order
- Resumes from last translated chapter if interrupted
- Includes progress tracking and error handling
- Rate limiting to prevent API issues

## Notes

- Uses GPT-4 for high-quality translations
- Requires OpenAI API credits
- Includes 3-second delay between requests to respect API limits
