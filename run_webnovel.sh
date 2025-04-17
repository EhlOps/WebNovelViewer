#!/bin/bash

# Set the working directory to the script's location
cd "$(dirname "$0")"

# Create directories if they don't exist
echo "Creating directories if they don't exist..."
mkdir -p chapters
mkdir -p chapters_translated

# Create and set up virtual environment
echo "Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv .venv
fi

# Source the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Run the translation script
echo "Running translation script..."
python translate_novel.py

# Run the filename cleaning script
echo "Running filename cleaning script..."
python clean_filenames.py

# Start the server in the background
echo "Starting the server..."
python server.py &
SERVER_PID=$!

# Wait for the server to start
echo "Waiting for server to start..."
sleep 3

# Start ngrok in the background
echo "Starting ngrok..."
ngrok http --url $NGROK_URL 3333 &
NGROK_PID=$!

# Function to handle script termination
cleanup() {
    echo "Shutting down..."
    kill $SERVER_PID
    kill $NGROK_PID
    deactivate  # Deactivate the virtual environment
    exit 0
}

# Set up trap to catch termination signals
trap cleanup SIGINT SIGTERM

# Keep the script running
echo "All services are running. Press Ctrl+C to stop."
echo "Server is running on http://localhost:3333"
echo "Ngrok URL will be displayed in the ngrok terminal window."

# Wait for user to press Ctrl+C
wait 
