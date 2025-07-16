# Video Analysis & Generation Pipeline

This pipeline allows you to analyze a video, generate detailed prompts for each segment, and use the MiniMax Hailou API to generate new video segments and images.

## 1. Environment Setup

1. **Clone the repository and navigate to the `video_analysis` directory.**
2. **Install dependencies:**
   - Create a virtual environment (optional but recommended):
     ```sh
     python3 -m venv video_analysis_venv
     source video_analysis_venv/bin/activate
     ```
   - Install required packages:
     ```sh
     pip install -r requirements.txt
     ```
3. **Set up your API key:**
   - Create a `.env` file in the `video_analysis` directory with the following content:
     ```
     HAILOU_API_KEY=your_minimax_api_key_here
     ```

## 2. Video Analysis: Generate Prompts

Use `video_analyzer_ffmpeg_hailou.py` to analyze a video and generate timeline-based prompts for each segment.

### Example usage:
```sh
python video_analyzer_ffmpeg_hailou.py --input path/to/video.mp4 --output analysis_results.json --segment-duration 6 --director-mode
```
- `--input`: Path to your input video file.
- `--output`: Path to save the analysis JSON file.
- `--segment-duration`: Duration (in seconds) for each segment (default: 6).
- `--director-mode`: (Optional) Use director-style prompts for more detailed, timeline-based descriptions.

## 3. Automatic Image Prompt Generation

You can have the analyzer script automatically generate an image prompt for the first frame using the `--image-prompt-out` argument. This uses Gemini (or another LLM) to analyze a frame and create a suitable prompt for image generation.

**Example usage:**
```sh
python video_analyzer_ffmpeg_hailou.py --input path/to/video.mp4 --output analysis_results.json --image-prompt-out rainbow_image_prompt.txt
```
- `--image-prompt-out rainbow_image_prompt.txt`: The script will generate an image prompt and save it to this file.

You can then use the generated prompt file in the image generation step:
```sh
python hailou_video_generator.py analysis_results.json --num-segments 1 --subject-ref path/to/subject.png --image-prompt rainbow_image_prompt.txt --image-out generated_image.png --no-video
```

This allows you to automate the creation of a high-quality image prompt for the first frame, without writing it manually.

## 4. Image Generation (First Frame)

To generate an image for the first frame using a prompt and a subject reference image:

### Example usage:
```sh
python hailou_video_generator.py analysis_results.json --num-segments 1 --subject-ref path/to/subject.png --image-prompt path/to/prompt.txt --image-out generated_image.png --no-video
```
- `--image-prompt`: Path to a text file containing your image prompt (2-3 sentences, <1200 chars recommended).
- `--subject-ref`: Path to the subject reference image (or a URL).
- `--image-out`: (Optional) Output path for the generated image.
- `--no-video`: Only generate the image, do not run video generation.

## 5. Video Generation

To generate videos for each segment using the analysis results and the MiniMax Hailou API:

### Example usage:
```sh
python hailou_video_generator.py analysis_results.json --num-segments 3 --subject-ref path/to/subject.png --output-dir generated_videos
```
- `analysis_results.json`: Output from the analyzer script.
- `--num-segments`: Number of segments to generate (required).
- `--subject-ref`: Path to the subject reference image (for the first segment).
- `--output-dir`: (Optional) Directory to save generated videos (default: `generated_videos`).

If you generated an image for the first frame, you can use it as the subject reference for video generation:
```sh
python hailou_video_generator.py analysis_results.json --num-segments 3 --subject-ref generated_image.png --output-dir generated_videos
```

## 6. Tips for Prompt Creation
- **Image prompts:** Use 2-3 sentences focusing on actions, environment, and context. Do **not** mention appearance, clothing, or facial features.
- **Video prompts:** Use the analyzer script's output, which is already formatted for the Hailou API.
- **Prompt length:** Keep prompts concise (ideally <1200 characters) to avoid API errors.

## 7. Output
- Generated videos and images will be saved in the specified output directory.
- A `generation_summary.json` file will summarize the results.

## 8. Troubleshooting
- If you see errors about prompt length, shorten your prompt.
- Ensure your API key is valid and has sufficient quota.
- Check that all file paths are correct and accessible.

---

For further customization or troubleshooting, see comments in the scripts or contact the maintainer. 