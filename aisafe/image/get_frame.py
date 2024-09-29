"""
author: @Cauch-BS
date: 2024-09-15

This script extracts video segments from the IEMOCAP dataset based on the timing information in the lab files.
The extracted video segments are saved as individual video files in the interim folder.
The IEMOCAP dataset is a multimodal dataset that contains audio, video, and text data for emotion recognition.

Functions:
- list_files_in_folder(folder_path): List all files in a folder.
- extract_video_segments(video_path, lab_file, output_folder): Extract video segments from a video file based on the timing information in a lab file.
"""

import os

import cv2
from tqdm import tqdm


def list_files_in_folder(folder_path: str) -> list:
  """
  List all files in a folder.

  Args:
    folder_path (str): The path to the folder.
  Returns:
    list: A list of file names in the folder.
  """
  # Get all items in the folder
  items: list[str] = os.listdir(folder_path)

  # Filter out directories, keeping only files
  files: list[str] = [
    item for item in items if os.path.isfile(os.path.join(folder_path, item))
  ]

  return files


def extract_video_segments(video_path: str, lab_file: str, output_folder: str) -> None:
  """
  Extract video segments from a video file based on the timing information in a lab file.

  Args:
    video_path (str): The path to the video file.
    lab_file (str): The path to the lab file containing the timing information.
    output_folder (str): The path to the folder to save the extracted video segments.

  Raises:
    FileNotFoundError: If the video file is not found.
    ValueError: If no frames are extracted for an utterance.

  Returns:
    None
  """
  # Open video file
  if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)

  # Read utterance timing from lab file
  with open(lab_file, "r") as f:
    lines = f.readlines()

  for line in lines:
    # Parse start and end time, and utterance name
    start_time, end_time, utterance_name = line.split()[:3]
    start_frame = int(float(start_time) * fps)
    end_frame = int(float(end_time) * fps)

    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read and save frames for the utterance
    frames = []
    for _ in range(start_frame, end_frame + 1):
      ret, frame = cap.read()
      if not ret:
        break
      frames.append(frame)

    if not frames:
      raise ValueError(f"No frames extracted for utterance {utterance_name}")

    # Save the extracted frames as a video clip
    out = cv2.VideoWriter(
      f"{output_folder}/{utterance_name}.mp4",
      cv2.VideoWriter_fourcc(*"avc1"),
      fps,
      (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
      out.write(frame)

    out.release()

  cap.release()


def main() -> None:
  avi_dir_list = [
    f"data/raw/iemocap/IEMOCAP_full_release/Session{i}/dialog/avi/DivX/"
    for i in range(1, 6)
  ]
  lab_dir_list = [
    f"data/raw/iemocap/IEMOCAP_full_release/Session{i}/dialog/lab/Ses0{i}_F/"
    for i in range(1, 6)
  ]

  for i, directory_pair in enumerate(zip(avi_dir_list, lab_dir_list)):
    video_dir, lab_dir = directory_pair
    files = list_files_in_folder(video_dir)
    for file in tqdm(files):
      if file.split(".")[-1] != "avi":
        continue
      video_path = video_dir + file
      lab_file = lab_dir + f"{file.split('.')[0]}.lab"
      output_folder = f"../data/interim/iemocap/Session{i + 1}"
      os.makedirs(output_folder, exist_ok=True)
      try:
        extract_video_segments(video_path, lab_file, output_folder)
      except Exception as e:
        print(f"Error processing {file}: {e}")
        print(f"Error occured at {video_path}, {lab_file}")
        break


if __name__ == "__main__":
  main()
