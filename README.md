# Web Novel Viewer

A web-based application for reading translated web novels with a responsive design that works on desktop and mobile devices.

## Features

- Responsive design that works on desktop, tablet, and mobile devices
- Dark mode support
- Collapsible sidebar for better mobile experience
- Chapter navigation with previous/next buttons
- Automatic chapter sorting
- Support for translated novel files

## Prerequisites

- Python 3.6 or higher
- ngrok (for external access)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/WebNovelViewer.git
   cd WebNovelViewer
   ```

2. Ensure the run script is executable:

   ```
   chmod +x run_webnovel.sh
   ```

## Usage

### Quick Start

Run the application using the provided script:

```
export OPENAI_API_KEY=your_api_key
export NGROK_URL=your_url
./run_webnovel.sh
```

This script will:

1. Create necessary directories (`chapters` and `chapters_translated`)
2. Set up a Python virtual environment (`.venv`)
3. Install required packages from `requirements.txt`
4. Run the translation script (`translate_novel.py`)
5. Run the filename cleaning script (`clean_filenames.py`)
6. Start the Flask server on port 3333
7. Start ngrok to provide external access

### Manual Setup

If you prefer to run the components manually:

1. Create and activate the virtual environment:

   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the translation script:

   ```
   python translate_novel.py
   ```

4. Run the filename cleaning script:

   ```
   python clean_filenames.py
   ```

5. Start the server:

   ```
   python server.py
   ```

6. In a separate terminal, start ngrok:
   ```
   ngrok http 3333
   ```

## Accessing the Application

- Local access: http://localhost:3333
- External access: Use the ngrok URL displayed in the terminal

## Directory Structure

- `chapters/`: Contains the original chapter files
- `chapters_translated/`: Contains the translated chapter files
- `translate_novel.py`: Script for translating novel chapters
- `clean_filenames.py`: Script for cleaning chapter filenames
- `server.py`: Flask server for serving the web application
- `index.html`: Main HTML file for the web interface
- `run_webnovel.sh`: Script to run all components

## Customization

### Adding New Chapters

1. Place the original chapter files in the `chapters/` directory
2. Run the translation script to generate translated versions
3. The web interface will automatically detect and display the new chapters

### Modifying the UI

The UI can be customized by editing the `index.html` file. The application uses CSS variables for theming, making it easy to change colors and dimensions.

## Troubleshooting

- **Server not starting**: Ensure port 3333 is not in use by another application
- **Chapters not appearing**: Check that the files are in the correct directory with the proper naming format
- **Translation issues**: Verify that the translation script has the necessary permissions and dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.
