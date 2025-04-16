from flask import Flask, send_file, jsonify, request, render_template_string
import os
import re

app = Flask(__name__)


# Serve the main HTML file with initial chapter state
@app.route("/")
@app.route("/chapter/<int:chapter_index>")
def index(chapter_index=None):
    with open("index.html", "r", encoding="utf-8") as f:
        content = f.read()

    # Add initial chapter state to the page
    initial_state = f"<script>window.initialChapterIndex = {chapter_index if chapter_index is not None else -1};</script>"
    content = content.replace("</head>", f"{initial_state}</head>")

    return content


# List all chapters
@app.route("/list-chapters")
def list_chapters():
    chapters = []
    for filename in os.listdir("chapters_translated"):
        if filename.endswith("_EN.txt"):
            # Extract only the first sequence of digits for sorting
            match = re.search(r"\d+", filename)
            if match:
                sort_key = int(match.group())
            else:
                # If no numeric part found, use the full filename
                sort_key = float("inf")

            # Remove the "_EN.txt" suffix from the display name
            display_name = filename[:-7]  # Remove last 7 characters ("_EN.txt")

            chapters.append(
                {"file": filename, "display": display_name, "sort_key": sort_key}
            )

    # Sort chapters by the numeric portion
    chapters.sort(key=lambda x: x["sort_key"])
    return jsonify(chapters)


# Read a specific chapter
@app.route("/read-chapter")
def read_chapter():
    chapter = request.args.get("chapter")
    if not chapter:
        return "Chapter not specified", 400

    chapter_path = os.path.join("chapters_translated", chapter)
    if not os.path.exists(chapter_path):
        return "Chapter not found", 404

    with open(chapter_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3333)
