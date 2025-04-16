import os
import re
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("filename_cleanup.log"), logging.StreamHandler()],
)


def clean_filename(filename):
    """
    Clean a filename by removing characters between the second hyphen and the first alphabetic character.

    Example:
    Input: "Chapter - 001 - Some Title_EN.txt"
    Output: "Chapter - 001 - Title_EN.txt"
    """
    # Match the pattern: "Chapter - number - [non-alphabetic characters]alphabetic character"
    pattern = r"(Chapter - \d+ - )[^a-zA-Z]*(.*)"
    match = re.match(pattern, filename)

    if match:
        prefix = match.group(1)  # "Chapter - number - "
        rest = match.group(2)  # The part after any non-alphabetic characters

        # Find the first alphabetic character in the rest of the string
        alpha_match = re.search(r"[a-zA-Z]", rest)
        if alpha_match:
            # Get everything from the first alphabetic character onwards
            cleaned_rest = rest[alpha_match.start() :]
            return f"{prefix}{cleaned_rest}"

    # If no match or no alphabetic character found, return the original filename
    return filename


def process_directory(directory_path):
    """
    Process all files in the given directory and rename them according to the cleaning rules.
    """
    directory = Path(directory_path)

    if not directory.exists():
        logging.error(f"Directory not found: {directory}")
        return

    # Get all files in the directory
    files = list(directory.glob("*.txt"))

    if not files:
        logging.info(f"No .txt files found in {directory}")
        return

    logging.info(f"Found {len(files)} files to process")

    # Process each file
    for file_path in files:
        original_name = file_path.name
        cleaned_name = clean_filename(original_name)

        if original_name != cleaned_name:
            new_path = file_path.parent / cleaned_name

            # Check if the new filename already exists
            if new_path.exists():
                logging.warning(
                    f"Cannot rename {original_name} to {cleaned_name} - file already exists"
                )
                continue

            try:
                # Rename the file
                file_path.rename(new_path)
                logging.info(f"Renamed: {original_name} -> {cleaned_name}")
            except Exception as e:
                logging.error(f"Error renaming {original_name}: {str(e)}")
        else:
            logging.info(f"No changes needed for: {original_name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean filenames by removing characters between the second hyphen and the first alphabetic character"
    )
    parser.add_argument("directory", help="Directory containing files to process")
    args = parser.parse_args()

    process_directory(args.directory)
    logging.info("Filename cleanup complete")


if __name__ == "__main__":
    main()
