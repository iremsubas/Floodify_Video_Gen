import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: analyze video, extract first frame, generate video/image.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--analysis-json", default="analysis.json", help="Path to output analysis JSON")
    parser.add_argument("--image-prompt", default="image_prompt.txt", help="Path to output image prompt text file")
    parser.add_argument("--output-dir", default="generated_videos", help="Output directory")
    parser.add_argument("--num-segments", type=int, default=1, help="Number of segments to generate")
    parser.add_argument("--hailou-api-key", default=None, help="Hailou API key")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key")
    parser.add_argument("--image-size", default="1024x1536", help="Image size for gpt-image-1")
    parser.add_argument("--no-video", action="store_true", help="Only generate the image, not the video")
    parser.add_argument("--first-frame-path", default="first_frame.png", help="Where to save the extracted first frame")
    parser.add_argument("--segment-duration", type=int, default=10, help="Segment duration for analyzer")
    parser.add_argument("--analyzer-model", default="gemini-2.5-pro", help="Model for analyzer")
    parser.add_argument("--google-api-key", default=None, help="Google API key for analyzer")
    parser.add_argument("--character-ref", help="Path to character reference image (for image generation, first image)")
    args = parser.parse_args()

    # Step 1: Run the analyzer
    analyzer_cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "video_analyzer_ffmpeg_hailou.py"),
        args.video,
        "--segment-duration", str(args.segment_duration),
        "--model", args.analyzer_model,
        "--output", args.analysis_json,
        "--image-prompt-out", args.image_prompt
    ]
    if args.google_api_key:
        analyzer_cmd += ["--api-key", args.google_api_key]
    print("Running analyzer:", " ".join(analyzer_cmd))
    subprocess.run(analyzer_cmd, check=True)

    # Step 2: Extract first frame
    print(f"Extracting first frame from {args.video} to {args.first_frame_path}...")
    subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), "extract_first_frame.py"),
        args.video, args.first_frame_path
    ], check=True)

    # Step 3: Call the image/video generation script
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "hailou_video_generator.py"),
        args.analysis_json,
        "--image-prompt", args.image_prompt,
        "--subject-ref", args.first_frame_path,
        "--output-dir", args.output_dir,
        "--num-segments", str(args.num_segments),
        "--image-size", args.image_size,
        "--first-frame", args.first_frame_path
    ]
    if args.hailou_api_key:
        cmd += ["--api-key", args.hailou_api_key]
    if args.openai_api_key:
        cmd += ["--openai-api-key", args.openai_api_key]
    if args.character_ref:
        cmd += ["--character-ref", args.character_ref]
    if args.no_video:
        cmd.append("--no-video")

    print("Running video/image generator:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Error running hailou_video_generator.py")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main() 