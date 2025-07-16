import os
import subprocess
import tempfile
import shutil
import google.generativeai as genai
from dotenv import load_dotenv
import json
import argparse
import time
import cv2
import base64

load_dotenv(dotenv_path=".env")

def get_video_duration(video_path):
    """Get video duration using FFmpeg."""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return float(result.stdout.strip())
    else:
        raise Exception("Failed to get video duration")

def split_video_with_ffmpeg(video_path, segment_duration=5):
    """Split video into segments using FFmpeg."""
    # Get actual video duration
    total_duration = get_video_duration(video_path)
    print(f"Video duration: {total_duration:.2f} seconds")
    
    # Create temporary directory for segments
    temp_dir = tempfile.mkdtemp()
    
    # Calculate number of segments needed
    num_segments = int((total_duration + segment_duration - 1) // segment_duration)
    print(f"Creating {num_segments} segments of {segment_duration} seconds each")
    
    segment_files = []
    
    for i in range(num_segments):
        start_time = i * segment_duration
        output_file = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
        
        # For the last segment, go to the end of the video
        if i == num_segments - 1:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-ss", str(start_time),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                output_file
            ]
        else:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-ss", str(start_time),
                "-t", str(segment_duration),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                output_file
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            segment_files.append(output_file)
            print(f"Created segment {i+1}: {start_time:.2f}s - {min(start_time + segment_duration, total_duration):.2f}s")
        else:
            print(f"Error creating segment {i+1}: {result.stderr}")
    
    print(f"Created {len(segment_files)} segments")
    return temp_dir, segment_files

def wait_for_file_active(uploaded_file, poll_interval=5, timeout=300):
    """Wait for uploaded file to become active."""
    start_time = time.time()
    while True:
        file_status = genai.get_file(uploaded_file.name)
        print(f"File status: {file_status.state}")
        if file_status.state == 2 or file_status.state == "ACTIVE":
            return file_status
        if time.time() - start_time > timeout:
            raise TimeoutError("File did not become ACTIVE in time.")
        time.sleep(poll_interval)

def analyze_video_segment(model, uploaded_file, segment_index, start_time, end_time, prompt_template):
    """Analyze a single video segment."""
    segment_duration = end_time - start_time if end_time else None
    
    prompt = prompt_template.format(
        start=start_time,
        end=end_time if end_time else "end",
        segment_index=segment_index,
        segment_duration=segment_duration if segment_duration else "variable"
    )
    
    contents = [
        uploaded_file,
        prompt
    ]
    
    response = model.generate_content(contents)
    return response.text.strip()

def extract_first_n_frames(video_path, n=3, output_dir=None):
    """Extract the first n frames from a video and return their file paths."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    count = 0
    while count < n:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"first_frame_{count:02d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        count += 1
    cap.release()
    return frame_paths

def analyze_frames_for_image_prompt(frame_paths, model, api_key):
    """Analyze the first N frames and generate a prompt for an image generation model."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model)
    descriptions = []
    for i, frame_path in enumerate(frame_paths):
        with open(frame_path, "rb") as img_file:
            image_bytes = img_file.read()
        prompt = (
    """Describe all visible actions, motions, gestures, background, and environmental details in this image.

**Instructions:**
- Focus entirely on what is happening in the scene and the context.
- Include details about:
  - Actions or motions being performed
  - Gestures or interactions
  - The background, setting, and environment (e.g., location, lighting, weather, objects present)
- **Do NOT mention or describe the subject’s physical appearance, clothing, facial features, or body type.**
- Do NOT speculate about identity, age, gender, or style.
- Write the description as a clear, concise prompt suitable for an image generation model.
- **Write your description in 2 to 3 sentences.**

**Example:**
"A person is jumping over a puddle on a city street at dusk, with blurred car lights in the background and wet pavement reflecting neon signs. The scene is lively and energetic, with rain still falling and umbrellas visible in the distance."

Begin your description below:
"""
        )
        response = model.generate_content([
            {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
            {"text": prompt}
        ])
        descriptions.append(response.text.strip())
    # Combine descriptions into a single prompt
    combined = " ".join(descriptions)
    image_prompt = f"Create an image with these characteristics: {combined}"
    return image_prompt

def analyze_video_with_ffmpeg_segments(video_path, api_key, prompt_template, segment_duration=5, model_name="gemini-2.5-pro", output_file=None):
    """Analyze video by splitting it into segments with FFmpeg."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Get video duration for accurate timing
    total_duration = get_video_duration(video_path)
    
    # Split video into segments
    temp_dir, segment_files = split_video_with_ffmpeg(video_path, segment_duration)
    
    results = []
    
    try:
        for i, segment_file in enumerate(segment_files):
            print(f"\nAnalyzing segment {i+1}/{len(segment_files)}...")
            
            # Calculate actual start and end times for this segment
            start_time = i * segment_duration
            if i == len(segment_files) - 1:
                # Last segment - go to end of video
                end_time = total_duration
            else:
                end_time = start_time + segment_duration
            
            # Upload segment to Gemini
            uploaded_file = genai.upload_file(segment_file)
            uploaded_file = wait_for_file_active(uploaded_file)
            
            # Analyze segment
            segment_text = analyze_video_segment(
                model, uploaded_file, i, start_time, end_time, prompt_template
            )
            
            result = {
                "segment_index": i,
                "start": start_time,
                "end": end_time,
                "summary": segment_text
            }
            
            results.append(result)
            print(f"[{start_time:.2f}s - {end_time:.2f}s]:\n{segment_text}\n{'-'*60}")
            
            # Small delay to respect rate limits
            time.sleep(1)
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze video segments using FFmpeg and Gemini API.")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--api-key", help="Google API key (or set GOOGLE_API_KEY in env)")
    parser.add_argument("--segment-duration", type=int, default=5, help="Segment duration in seconds (default: 5)")
    parser.add_argument("--director-mode", action="store_true", help="Use director mode with enhanced camera control prompts")
    parser.add_argument(
        "--prompt",
        help="Prompt template for each segment. Use {start}, {end}, and {segment_index} for placeholders."
    )
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model to use")
    parser.add_argument("--output", "-o", help="Output file path (JSON)")
    parser.add_argument("--image-prompt-out", help="Output file for generated image prompt from first frames")
    args = parser.parse_args()

    # Set default prompt based on director mode
    if not args.prompt:
        if args.director_mode:
            args.prompt = (
                """[Director Prompt]
Generate a highly detailed, photorealistic video sequence for a video generation model with director controls, based on this {segment_duration}-second segment (from {start:.2f}s to {end:.2f}s).

For each distinct shot within this segment, provide a precise timeline entry using the format:
[Start Time]-[End Time]: [Camera Movement(s), Shot Type] [Detailed description of action, subjects, and environment, including visual style and lighting]

**Key Directives for Hailou Model:**

* **Camera Movements:** Use up to three explicit camera movement commands in square brackets for each shot. Choose from:
    * `[Pan left]` / `[Pan right]` (camera pivots horizontally)
    * `[Tilt up]` / `[Tilt down]` (camera pivots vertically)
    * `[Zoom in]` / `[Zoom out]` (focal length change)
    * `[Truck left]` / `[Truck right]` (camera moves horizontally, parallel to subject)
    * `[Push in]` / `[Pull out]` (camera moves closer/further from subject)
    * `[Pedestal up]` / `[Pedestal down]` (camera moves vertically)
    * `[Tracking shot]` (camera follows a moving subject)
    * `[Static shot]` (camera remains stationary)
    * `[Cut]` (indicates a sharp transition to a new shot)
    * `[Shake]` (simulates handheld camera instability)
* **Shot Types:** Specify classic shot types like `Wide shot`, `Medium shot`, `Close-up`, `Extreme close-up`, `Aerial shot`, `Over-the-shoulder`, `Point-of-view (POV)`.
* **Detail and Specificity:**
    * Describe *exactly* what is visible and what is happening.
    * Include **specific actions** and **movements** of subjects.
    * Detail the **environment and setting** (e.g., "lush green forest," "desert with red rock formations," "bustling city street at night").
    * Mention **landmarks** if present (e.g., "Eiffel Tower in the background," "Christ the Redeemer statue").
    * Clearly indicate **scene changes or transitions** within the segment.
    * Specify **lighting and atmosphere**: "golden hour," "dramatic backlighting," "soft, diffused light," "neon reflections," "overcast sky."
    * Include **visual style elements** if applicable, e.g., "cinematic," "photorealistic," "dreamy," "gritty."
* **Exclusions:** Do NOT mention clothing, appearance, or physical looks of subjects. Focus solely on their actions and the visual scene.
* **Conciseness:** Each shot description should be 1-2 concise, impactful sentences, packed with visual information.

**Example Timeline Entries:**

0–2s: [Wide shot, Pan left] A helicopter cockpit with a pilot speaking into a headset, city skyline visible through the window, illuminated by the warm glow of sunset.
2–4s: [Cut, Aerial shot, Zoom in] Camera flies over lush green mountains, revealing the Christ the Redeemer statue atop a peak under a clear blue sky, casting a long shadow.
4–6s: [Tracking shot, Pedestal up] Camera circles the statue, city and ocean in the background, with the sunlight glinting off the statue's surface."""
            )
        else:
            args.prompt = (
#                 """Analyze this {segment_duration}-second video segment (from {start:.2f}s to {end:.2f}s) and create a detailed description for video generation.

# Describe what happens in this segment with the following focus:
# 1. **Camera movements and shot types** (pan, tilt, zoom, tracking, static, wide shot, close-up, etc.)
# 2. **Subjects and their actions** (what people or objects are doing)
# 3. **Environment and setting** (background, location, lighting, atmosphere)
# 4. **Scene changes or transitions** (cuts, fades, etc.)

# Be specific about:
# - Camera work: "pan left", "zoom in", "tracking shot", "static shot"
# - Actions: "speaking into microphone", "walking through doorway", "dancing energetically"
# - Environment: "brightly lit room", "dimly lit hallway", "outdoor setting"
# - Lighting: "soft lighting", "purple ambient light", "natural daylight"

# Format your response as a clear, detailed description suitable for video generation.
# Keep it concise but informative - enough detail for AI video generation without being overwhelming.
# Do not mention clothing or appearance - focus on actions, context, and camera work."""
                # """
                # Focus on: 
                # 1. What subjects or objects are visible and what they are doing 
                # 2. Any scene changes, cuts, or transitions that occur within this segment 
                # 3. Camera movements (pan, tilt, zoom, tracking, stationary) and their direction 
                # 4. Background environment and setting details 
                # 5. Any actions, movements, or events that take place 
                # Be specific about timing within the segment (e.g., 'at 2 seconds into this segment...'). 
                # If there are multiple scenes or cuts within this segment, describe each one separately with their timing. 
                # Do not include any description of the subject's appearance, clothing, or physical looks. 
                # Focus on what is happening, the context, and the camera work, not how the subject looks. 
                # Write this as a clear, detailed description suitable for video generation.
                # """
            """Analyze this {segment_duration}-second video segment (from {start:.2f}s to {end:.2f}s) and create a detailed timeline description for video generation.
            
            Format your response as a timeline with specific time ranges and shot descriptions, like this example:
            "0–2s: Wide shot of a dark ocean horizon under a starry dawn sky."
            "2–4s: The sun's rim appears, coloring clouds pastel pink."
            "4–6s: Golden light ripples on gentle waves; a seagull flies across."
            "6–8s: Close-up pan on a silhouette of a sailboat drifting; sky fades to bright blue."
            
            For each time range, include:
            1. Camera shot type and movement (wide shot, close-up, pan, tilt, zoom, tracking, dolly, etc.)
            2. Subjects and their actions (what people/objects are doing)
            3. Environment and setting details (landmarks, buildings, landscape, weather)
            4. Any scene transitions or cuts
            
            Be specific about:
            - Camera movements: "pan left", "zoom in", "tracking shot", "static shot"
            - Actions: "speaking into microphone", "flying over", "circling around"
            - Landmarks: "Christ the Redeemer statue", "mountain peak", "city skyline"
            - Environment: "lush green forest", "blue sky with clouds", "helicopter cockpit"
            
            Use this exact format: "0–2s: [detailed shot description with camera movement and action]"
            Keep descriptions concise but informative - enough detail for video generation without being overwhelming.
            Do not mention clothing or appearance - focus on actions, context, and camera work.
            """
            )

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key is required.")

    # If --image-prompt-out is provided, analyze first N frames and write image prompt
    if args.image_prompt_out:
        frame_paths = extract_first_n_frames(args.video_path, n=3)
        image_prompt = analyze_frames_for_image_prompt(frame_paths, args.model, api_key)
        with open(args.image_prompt_out, "w", encoding="utf-8") as f:
            f.write(image_prompt)
        print(f"Image prompt written to {args.image_prompt_out}")
        return

    analyze_video_with_ffmpeg_segments(
        args.video_path,
        api_key,
        args.prompt,
        segment_duration=args.segment_duration,
        model_name=args.model,
        output_file=args.output
    )

if __name__ == "__main__":
    main() 