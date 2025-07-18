# Video Analysis & Generation Pipeline

This pipeline allows you to analyze a video, generate detailed prompts for each segment, and use the MiniMax Hailou API to generate new video segments and images.

## 1. Automated End-to-End Pipeline (Recommended)

The easiest way to run the full workflow is with the `wrapper_pipeline.py` script. This script will:
- Analyze your video and generate prompts for each segment
- Extract the first frame from the video
- Generate an image for the first frame using OpenAI (optionally with a character reference)
- Generate videos for each segment using the MiniMax Hailou API

### Example usage:
```sh
python wrapper_pipeline.py \
  --video path/to/video.mp4 \
  --character-ref path/to/character.png \
  --google-api-key <your_google_key> \
  --hailou-api-key <your_hailou_key> \
  --openai-api-key <your_openai_key> \
  --num-segments 3
```
- `--video`: Path to your input video file
- `--character-ref`: (Optional) Path to a character reference image (used as the first image for image generation)
- `--num-segments`: Number of segments to generate (required)
- `--google-api-key`, `--hailou-api-key`, `--openai-api-key`: Your API keys for Gemini, MiniMax Hailou, and OpenAI
- Other options: see `python wrapper_pipeline.py --help`

The wrapper will automatically:
- Run the analyzer to produce `analysis.json` and `image_prompt.txt`
- Extract the first frame to `first_frame.png`
- Call `hailou_video_generator.py` with both `--character-ref` and `--first-frame` for image generation
- Run video generation for all segments

## 2. Environment Setup

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
3. **Set up your API keys:**
   - Create a `.env` file in the `video_analysis` directory with the following content:
     ```
     HAILOU_API_KEY=your_minimax_api_key_here
     GOOGLE_API_KEY=your_gemini_or_google_api_key_here
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - If you add your Gemini/Google and OpenAI API keys to the `.env` file as shown above, you **do not need to specify them on the command line** when running the wrapper or other scripts—they will be loaded automatically.

## 3. Advanced/Manual Usage

You can also run each step individually for more control or debugging.

### 3.1 Video Analysis: Generate Prompts
Use `video_analyzer_ffmpeg_hailou.py` to analyze a video and generate timeline-based prompts for each segment.

**Example usage:**
```sh
python video_analyzer_ffmpeg_hailou.py path/to/video.mp4 --output analysis_results.json --segment-duration 6
```
- `path/to/video.mp4`: Path to your input video file.
- `--output`: Path to save the analysis JSON file.
- `--segment-duration`: Duration (in seconds) for each segment (default: 6).

### 3.2 Automatic Image Prompt Generation
You can have the analyzer script automatically generate an image prompt for the first frame using the `--image-prompt-out` argument. This uses Gemini (or another LLM) to analyze a frame and create a suitable prompt for image generation.

**Example usage:**
```sh
python video_analyzer_ffmpeg_hailou.py path/to/video.mp4 --output analysis_results.json --image-prompt-out rainbow_image_prompt.txt
```
- `--image-prompt-out rainbow_image_prompt.txt`: The script will generate an image prompt and save it to this file.

You can then use the generated prompt file in the image generation step:
```sh
python hailou_video_generator.py analysis_results.json --num-segments 1 --subject-ref path/to/subject.png --image-prompt rainbow_image_prompt.txt --image-out generated_image.png --no-video
```

### 3.3 Image Generation (First Frame)
To generate an image for the first frame using a prompt and a subject reference image:

**Example usage:**
```sh
python hailou_video_generator.py analysis_results.json --num-segments 1 --subject-ref path/to/subject.png --image-prompt path/to/prompt.txt --image-out generated_image.png --no-video
```
- `--image-prompt`: Path to a text file containing your image prompt (2-3 sentences, <1200 chars recommended).
- `--subject-ref`: Path to the subject reference image (or a URL).
- `--image-out`: (Optional) Output path for the generated image.
- `--no-video`: Only generate the image, do not run video generation.

### 3.4 Video Generation
To generate videos for each segment using the analysis results and the MiniMax Hailou API:

**Example usage:**
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

## 4. Tips for Prompt Creation
- **Image prompts:** Use 2-3 sentences focusing on actions, environment, and context. Do **not** mention appearance, clothing, or facial features.
- **Video prompts:** Use the analyzer script's output, which is already formatted for the Hailou API.
- **Prompt length:** Keep prompts concise (ideally <1200 characters) to avoid API errors.

## 5. Output
- Generated videos and images will be saved in the specified output directory.
- A `generation_summary.json` file will summarize the results.

## 6. Troubleshooting
- If you see errors about prompt length, shorten your prompt.
- Ensure your API key is valid and has sufficient quota.
- Check that all file paths are correct and accessible.

---

For further customization or troubleshooting, see comments in the scripts or contact the maintainer. 