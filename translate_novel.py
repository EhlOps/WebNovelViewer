import os
import time
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import openai
import httpx
from tqdm import tqdm
from contextlib import contextmanager
import signal
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import threading
import backoff  # Add this for exponential backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(message)s",
    handlers=[logging.FileHandler("translation.log"), logging.StreamHandler()],
)


class TimeoutException(Exception):
    pass


class FormatErrorException(Exception):
    """Exception raised when the API response format is invalid."""

    pass


@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("API call timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)


class ChapterTranslator:
    def __init__(self, api_key: str, output_dir: Path):
        self.client = openai.OpenAI(
            api_key=api_key,
            timeout=60.0,
            max_retries=0,
        )
        self.output_dir = output_dir
        self.lock = threading.Lock()

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIError, TimeoutException),
        max_tries=10,  # Maximum number of retries
        max_time=300,  # Maximum total time to try (5 minutes)
    )
    def translate_with_retry(self, messages: List[Dict[str, str]]) -> Dict:
        """Make API call with exponential backoff retry"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            # Validate that we got a response with content
            if not response.choices or not response.choices[0].message.content:
                raise FormatErrorException("Empty response from API")

            return response
        except Exception as e:
            if isinstance(e, FormatErrorException):
                # Don't retry format errors, they're likely to repeat
                raise
            # For other errors, let the backoff mechanism handle retries
            raise

    def translate_single_chapter(self, chapter_path: Path) -> bool:
        """Translate a single chapter with unlimited retries until success."""
        chapter_num, korean_title = self.parse_chapter_info(chapter_path.name)

        # Check if already translated
        existing_translations = list(
            self.output_dir.glob(f"Chapter - {chapter_num:04d}*_EN.txt")
        )
        if existing_translations:
            logging.info(f"Skipping {chapter_path.name} - already translated")
            return True

        retry_count = 0
        max_retries = 3  # Limit the number of retries for format issues

        while True:  # Keep trying until successful
            try:
                logging.info(f"Attempting translation of chapter {chapter_num}")

                chapter_text = self.read_chapter(chapter_path)
                if not chapter_text:
                    logging.error(f"Failed to read chapter {chapter_path}")
                    time.sleep(5)  # Wait before retry
                    continue

                result = self.translate_chapter(chapter_text, korean_title)
                if not result:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(
                            f"Failed to translate chapter {chapter_num} after {max_retries} attempts. Skipping."
                        )
                        return False

                    logging.error(
                        f"Translation failed for chapter {chapter_num}, retrying... (Attempt {retry_count}/{max_retries})"
                    )
                    time.sleep(5)  # Wait before retry
                    continue

                if self.save_translation(
                    chapter_path, result["title"], result["content"]
                ):
                    logging.info(f"Successfully translated chapter {chapter_num}")
                    return True
                else:
                    logging.error(
                        f"Failed to save translation for chapter {chapter_num}, retrying..."
                    )
                    time.sleep(5)  # Wait before retry
                    continue

            except FormatErrorException as e:
                logging.error(f"Format error for chapter {chapter_num}: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(
                        f"Failed to get valid format for chapter {chapter_num} after {max_retries} attempts. Skipping."
                    )
                    return False
                time.sleep(5)  # Wait before retry
                continue
            except Exception as e:
                logging.error(f"Error processing chapter {chapter_num}: {str(e)}")
                time.sleep(5)  # Wait before retry
                continue

    def parse_chapter_info(self, filename: str) -> Tuple[int, str]:
        match = re.match(r"Chapter - (\d+) - (.+?)\.txt$", filename)
        if match:
            return int(match.group(1)), match.group(2)
        raise ValueError(f"Invalid chapter filename format: {filename}")

    def read_chapter(self, chapter_path: Path) -> Optional[str]:
        try:
            with open(chapter_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading chapter {chapter_path}: {str(e)}")
            return None

    def translate_chapter(
        self, chapter_text: str, chapter_title: str
    ) -> Optional[Dict[str, str]]:
        """Translate a chapter with retries."""
        max_chars = 12000
        if len(chapter_text) > max_chars:
            chapter_text = chapter_text[:max_chars]

        # Extract title from the first line if it exists
        lines = chapter_text.strip().split("\n")
        extracted_title = lines[0] if lines else chapter_title
        content_to_translate = "\n".join(lines[1:]) if len(lines) > 1 else chapter_text

        messages = [
            {
                "role": "system",
                "content": "You are a professional Korean to English novel translator. You must always respond with a valid JSON object containing exactly the required fields.",
            },
            {
                "role": "user",
                "content": f"""Translate this Korean novel chapter.
Title: {extracted_title}

Content:
{content_to_translate}

Notes:
Shirone is a boy. Amy is a girl. The thing that swordfighters use is called a Schema. The thing that mages use is called a Spirit Zone.

Return a JSON object with the following fields:
1. 'title': The English translation of the title
2. 'content': The English translation of the content

Your response must be a single JSON object with no additional text or formatting.
""",
            },
        ]

        try:
            response = self.translate_with_retry(messages)
            content = response.choices[0].message.content

            # Try to parse the JSON response
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {str(e)}")
                logging.error(f"Raw response: {content}")
                return None

            # Validate response format with strict schema
            required_fields = ["title", "content"]
            missing_fields = [field for field in required_fields if field not in result]

            if missing_fields:
                logging.error(f"Missing required fields: {missing_fields}")
                logging.error(f"Response: {result}")
                return None

            # Validate field types
            for field in required_fields:
                if not isinstance(result[field], str):
                    logging.error(
                        f"Field '{field}' is not a string: {type(result[field])}"
                    )
                    return None

            # Validate field content
            if not result["title"].strip():
                logging.error("Title is empty")
                return None

            if not result["content"].strip():
                logging.error("Content is empty")
                return None

            return result

        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return None

    def save_translation(
        self, chapter_path: Path, translated_title: str, translated_text: str
    ) -> bool:
        try:
            chapter_num, _ = self.parse_chapter_info(chapter_path.name)
            output_filename = f"Chapter - {chapter_num:04d} - {translated_title}_EN.txt"
            output_path = self.output_dir / output_filename

            with self.lock:  # Thread-safe file writing
                with open(output_path, "w", encoding="utf-8") as f:
                    # Write the translated title as the first line
                    f.write(f"{translated_title}\n\n")
                    # Write the translated content
                    f.write(translated_text)
            return True
        except Exception as e:
            logging.error(f"Error saving translation for {chapter_path}: {str(e)}")
            return False


def translate_chapter_wrapper(args: Tuple[Path, str, Path]) -> bool:
    """Wrapper function for multiprocessing that ensures completion"""
    chapter_path, api_key, output_dir = args
    translator = ChapterTranslator(api_key, output_dir)

    try:
        result = translator.translate_single_chapter(chapter_path)
        return result
    except Exception as e:
        logging.error(f"Error in translation wrapper: {str(e)}")
        return False


class NovelTranslator:
    def __init__(
        self,
        api_key: str,
        source_dir: str,
        output_dir: str,
        start_chapter: int = 1,
        num_processes: int = None,
    ):
        self.api_key = api_key
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.start_chapter = start_chapter
        self.num_processes = num_processes or max(1, cpu_count() - 1)

    def get_sorted_chapters(self) -> List[Path]:
        try:
            chapters = list(self.source_dir.glob("Chapter - *.txt"))
            sorted_chapters = sorted(
                chapters,
                key=lambda x: int(re.search(r"Chapter - (\d+)", x.name).group(1)),
            )

            if not sorted_chapters:
                return []

            # Reorder chapters starting from start_chapter
            start_idx = 0
            for i, chapter in enumerate(sorted_chapters):
                chapter_num = int(re.search(r"Chapter - (\d+)", chapter.name).group(1))
                if chapter_num == self.start_chapter:
                    start_idx = i
                    break

            return sorted_chapters[start_idx:] + sorted_chapters[:start_idx]

        except Exception as e:
            logging.error(f"Error getting sorted chapters: {str(e)}")
            return []

    def translate_novel(self):
        chapters = self.get_sorted_chapters()
        if not chapters:
            logging.error("No chapters found to translate")
            return

        total_chapters = len(chapters)
        logging.info(f"Found {total_chapters} chapters to translate")
        logging.info(
            f"Starting from chapter {self.start_chapter} using {self.num_processes} processes"
        )

        # Prepare arguments for parallel processing
        args = [(chapter, self.api_key, self.output_dir) for chapter in chapters]

        # Create process pool and translate chapters in parallel
        with Pool(processes=self.num_processes) as pool:
            with tqdm(total=len(chapters), desc="Translating chapters") as pbar:
                for _ in pool.imap_unordered(translate_chapter_wrapper, args):
                    pbar.update(1)

        logging.info("Translation complete. All chapters processed successfully.")


def main():
    # Install required package
    os.system("pip install backoff")

    parser = argparse.ArgumentParser(
        description="Translate Korean novel chapters to English"
    )
    parser.add_argument(
        "--start-chapter",
        type=int,
        default=1,
        help="Chapter number to start translation from (default: 1)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of parallel processes to use (default: CPU count - 1)",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable not set")
        return

    try:
        translator = NovelTranslator(
            api_key=api_key,
            source_dir="chapters",
            output_dir="chapters_translated",
            start_chapter=args.start_chapter,
            num_processes=args.processes,
        )

        translator.translate_novel()

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()
