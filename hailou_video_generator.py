import os
import json
import requests
import time
import argparse
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import base64
import subprocess
import tempfile
import openai

"""
Hailou Video Generator
---------------------
This script takes the output JSON from video_analyzer_ffmpeg_hailou.py and generates videos for each segment using the MiniMax Hailou API.

Workflow:
1. For each segment, format the prompt for Hailou.
2. POST to https://api.minimax.io/v1/video_generation with required parameters.
3. Poll status at https://api.minimax.io/v1/query/video_generation?task_id={task_id}
4. When status is 'success', use the returned file_id to get the video URL via the File (Retrieve) API.
5. Download the video to the output directory.

Required parameters: model, prompt, duration, resolution, first_frame_image
"""

load_dotenv(dotenv_path=".env")

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 data URL."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        mime = 'image/jpeg'
    elif ext == '.png':
        mime = 'image/png'
    elif ext == '.webp':
        mime = 'image/webp'
    else:
        mime = 'image/jpeg'  # Default fallback
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

def encode_image_to_base64_data_url(image_path: str) -> str:
    """Encode image file to base64 data URL for OpenAI input_image."""
    import base64
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        mime = 'image/jpeg'
    elif ext == '.png':
        mime = 'image/png'
    elif ext == '.webp':
        mime = 'image/webp'
    else:
        mime = 'image/jpeg'  # Default fallback
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

class HailouVideoGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HAILOU_API_KEY")
        if not self.api_key:
            raise ValueError("Hailou API key is required. Set HAILOU_API_KEY in .env or pass --api-key")
        self.base_url = "https://api.minimax.io/v1/video_generation"
        self.status_url = "https://api.minimax.io/v1/query/video_generation"
        self.file_url = "https://api.minimax.io/v1/files/retrieve"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def poll_video_status(self, task_id: str, max_wait_time: int = 600) -> Dict[str, Any]:
        """Poll the status of a video generation task."""
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(
                    f"{self.status_url}?task_id={task_id}",
                    headers=self.headers,
                    timeout=10
                )
                if response.status_code != 200:
                    print(f"Status check error: {response.status_code}")
                    time.sleep(5)
                    continue
                status_data = response.json()
                status = status_data.get("status", "unknown").lower()
                print(f"Task {task_id} status: {status}")
                if status == "success":
                    return status_data
                elif status == "failed":
                    raise Exception(f"Video generation failed: {status_data}")
                elif status in ["processing", "pending"]:
                    time.sleep(10)
                else:
                    print(f"Unknown status: {status}")
                    time.sleep(5)
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                time.sleep(5)
        raise TimeoutError(f"Video generation timed out after {max_wait_time} seconds")

    def get_video_url_from_file_id(self, file_id: str, group_id: str = "1944913641618805059", retries: int = 5, delay: int = 10) -> str:
        """Get video URL from file ID using the File (Retrieve) API."""
        url = f"https://api.minimax.io/v1/files/retrieve?GroupId={group_id}&file_id={file_id}"
        for attempt in range(retries):
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                video_url = data.get("file", {}).get("download_url")
                if video_url:
                    return video_url
            print(f"File not ready (attempt {attempt+1}/{retries}), retrying in {delay}s...")
            time.sleep(delay)
        raise Exception(f"File API error: {response.status_code} - {response.text}")

    def download_video(self, video_url: str, output_path: str) -> str:
        """Download video from URL to local file."""
        print(f"Downloading video from {video_url}...")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded to: {output_path}")
        return output_path

def load_analysis_results(json_file: str) -> List[Dict[str, Any]]:
    """Load analysis results from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_prompt_for_hailou(segment_data: Dict[str, Any]) -> str:
    """Format segment data for Hailou API."""
    return segment_data["summary"].strip()

def extract_last_frame(video_path: str, output_image_path: str) -> None:
    """Extract the last frame of a video using ffmpeg and save it as output_image_path."""
    cmd = [
        "ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path, "-vframes", "1", output_image_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract last frame from {video_path}: {e.stderr.decode()}")

def upload_image_and_get_file_id(client, image_path: str) -> str:
    """Upload an image to OpenAI Files API and return the file_id."""
    with open(image_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id


def generate_image_with_gpt_image_1_responses(prompt_file: str, api_key: str, output_path: str = "generated_image.png", character_ref_image: Optional[str] = None, first_frame_image: Optional[str] = None):
    """Generate an image using OpenAI responses.create with image_generation tool, optionally using two images as base64 data URLs."""
    import base64
    import os
    client = openai.OpenAI(api_key=api_key)
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    content = [
        {"type": "input_text", "text": prompt}
    ]
    if character_ref_image:
        abs_path1 = os.path.abspath(character_ref_image)
        data_url1 = encode_image_to_base64_data_url(abs_path1)
        content.append({
            "type": "input_image",
            "image_url": data_url1
        })
    if first_frame_image:
        abs_path2 = os.path.abspath(first_frame_image)
        data_url2 = encode_image_to_base64_data_url(abs_path2)
        content.append({
            "type": "input_image",
            "image_url": data_url2
        })
    input_list = [
        {
            "role": "user",
            "content": content
        }
    ]
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=input_list,
        tools=[{"type": "image_generation"}]
    )
    # Extract base64 image from response
    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]
    if not image_data:
        raise Exception("No image data returned from OpenAI responses.create")
    image_base64 = image_data[0]
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    print(f"Image generated and saved to {output_path}")

def generate_videos_from_analysis(
    analysis_file: str,
    output_dir: str = "generated_videos",
    num_segments: Optional[int] = None,
    api_key: Optional[str] = None,
    subject_ref: Optional[str] = None
):
    """Generate videos from analysis results using Hailou API."""
    # Ensure output directory exists at the very start
    os.makedirs(output_dir, exist_ok=True)
    segments = load_analysis_results(analysis_file)
    if num_segments is not None:
        segments = segments[:num_segments]
    generator = HailouVideoGenerator(api_key)
    generated_videos = []
    
    for i, segment in enumerate(segments):
        # Ensure output directory exists before each segment
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Processing segment {i+1}/{len(segments)}")
        print(f"Time range: {segment['start']:.2f}s - {segment['end']:.2f}s")
        print(f"{'='*60}")
        
        prompt = format_prompt_for_hailou(segment)
        print(f"Formatted prompt: {prompt}")
        
        try:
            model = "MiniMax-Hailuo-02"
            duration = 10
            resolution = "768P"
            
            # For first segment, use the provided subject reference
            if i == 0:
                if not subject_ref:
                    raise ValueError("--subject-ref is required for the first segment")
                if os.path.isfile(subject_ref):
                    first_frame_image = encode_image_to_base64(subject_ref)
                else:
                    first_frame_image = subject_ref
            else:
                # For subsequent segments, extract last frame from previous video
                prev_video_path = generated_videos[-1]["video_path"]
                last_frame_path = os.path.join(output_dir, f"segment_{i-1:03d}_last_frame.png")
                extract_last_frame(prev_video_path, last_frame_path)
                with open(last_frame_path, "rb") as img_file:
                    first_frame_image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                first_frame_image = f"data:image/png;base64,{first_frame_image_b64}"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "prompt_optimizer": True,
                "first_frame_image": first_frame_image
            }
            
            print(f"Submitting video generation request...\nModel: {model}\nPrompt: {prompt[:100]}...\nDuration: {duration}s, Resolution: {resolution}, Optimizer: False")
            if i == 0:
                print("[DEBUG] Payload for first segment:")
                print(json.dumps(payload, indent=2)[:1000])
            
            response = requests.post(
                generator.base_url,
                headers=generator.headers,
                data=json.dumps(payload),
                timeout=30
            )
            if response.status_code != 200:
                raise Exception(f"Hailou API error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            task_id = response_data.get("task_id")
            if not task_id:
                print(f"Error: No task_id in response: {response_data}")
                continue
            
            print(f"Task ID: {task_id}")
            status_data = generator.poll_video_status(task_id)
            file_id = status_data.get("file_id")
            if not file_id:
                print(f"Error: No file_id in status response: {status_data}")
                continue
            
            video_url = generator.get_video_url_from_file_id(file_id)
            video_path = os.path.join(output_dir, f"segment_{i:03d}_{segment['start']}s_{segment['end']}s.mp4")
            generator.download_video(video_url, video_path)
            print(f"✅ Successfully generated video: {video_path}")
            
            generated_videos.append({
                "segment": i,
                "start": segment["start"],
                "end": segment["end"],
                "prompt": prompt,
                "video_path": video_path,
                "video_url": video_url,
                "file_id": file_id,
                "task_id": task_id
            })
            
            print("Waiting 5 seconds before next request...")
            time.sleep(5)
            
        except Exception as e:
            print(f"❌ Error generating video for segment {i+1}: {e}")
            continue
    
    summary_file = os.path.join(output_dir, "generation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(generated_videos, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Generated {len(generated_videos)} videos out of {len(segments)} segments")
    print(f"Results saved to: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"{'='*60}")
    
    return generated_videos

def main():
    parser = argparse.ArgumentParser(description="Generate videos from analysis results using Hailou API. For first frame image generation, uses OpenAI gpt-image-1.")
    parser.add_argument("analysis_file", help="Path to the analysis JSON file from video_analyzer_ffmpeg_hailou.py")
    parser.add_argument("--api-key", help="Hailou API key (or set HAILOU_API_KEY in .env)")
    parser.add_argument("--output-dir", default="generated_videos", help="Output directory for generated videos")
    parser.add_argument("--num-segments", type=int, required=True, help="Number of segments to generate")
    parser.add_argument("--subject-ref", required=True, help="Path to subject reference image (for first segment, only used for video)")
    parser.add_argument("--character-ref", help="Path to character reference image (for image generation, first image)")
    parser.add_argument("--first-frame", help="Path to first frame image (for image generation, second image)")
    parser.add_argument("--image-prompt", help="Path to prompt file for image generation (optional)")
    parser.add_argument("--image-out", default=None, help="Output path for generated image (optional)")
    parser.add_argument("--no-video", action="store_true", help="If set, only generate the image and do not run video generation.")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY in .env)")
    parser.add_argument("--image-size", default="1024x1536", help="Image size for gpt-image-1 (default: 1024x1536)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("HAILOU_API_KEY")
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Hailou API key is required.")

    # If --image-prompt is provided, generate the image first using OpenAI
    generated_image_path = None
    if args.image_prompt:
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for image generation.")
        if args.image_out:
            generated_image_path = args.image_out
        else:
            # Use a temp file if not provided
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            generated_image_path = tmp.name
            tmp.close()
        generate_image_with_gpt_image_1_responses(
            args.image_prompt, openai_api_key, generated_image_path,
            character_ref_image=args.character_ref, first_frame_image=args.first_frame
        )
        if args.no_video:
            return 0
        # Use the generated image as the subject_ref for video generation
        subject_ref_for_video = generated_image_path
    else:
        subject_ref_for_video = args.subject_ref

    try:
        generated_videos = generate_videos_from_analysis(
            args.analysis_file,
            args.output_dir,
            args.num_segments,
            api_key,
            subject_ref_for_video
        )
        print(f"\nSuccessfully generated {len(generated_videos)} videos!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 