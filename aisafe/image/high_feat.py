import os

import cv2
import numpy as np
import openface
from tqdm import tqdm


def list_files_in_folder(folder_path): #type: ignore
    # Get all items in the folder
    items = os.listdir(folder_path)

    # Filter out directories, keeping only files
    files = [item for item in items if os.path.isfile(os.path.join(folder_path, item))]

    return files

# Initialize the face alignment model
align = openface.AlignDlib("./models/shape_predictor_68_face_landmarks.dat")

# Initialize the OpenFace neural network
net = openface.TorchNeuralNet("./models/nn4.small2.v1.t7", imgDim=96, cuda = False)

def extract_action_units(image): #type: ignore
    # Convert image to RGB (OpenFace expects RGB format)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect and align the face
    bb = align.getLargestFaceBoundingBox(rgb_img)
    if bb is None:
        raise ValueError("No face detected")

    aligned_face = align.align(96, rgb_img, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    if aligned_face is None:
        raise ValueError("Failed to align face")

    # Extract face features using the neural network
    face_representation = net.forward(aligned_face)

    return face_representation

def extract_frames(video_path): #type: ignore
    # Open video file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
    frames = []

    start_frame = 0  # Define start_frame
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # Define end_frame as the last frame of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set video to start frame

    for i in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {i}")
            break
        frames.append(frame)

    cap.release()

    return frames

# Process frames to extract FAUs
def extract_faus_for_frames(frames): #type: ignore
    faus_per_frame = {}
    for i, frame in enumerate(frames):
        faus = extract_action_units(frame)
        faus_per_frame[f"frame_{i}"] = faus
    return faus_per_frame

# Usage
# video_path = './data/interim/iemocap/Session1/Ses01F_impro01_F000.mp4'
# frames = extract_frames(video_path)
# faus_per_frame = extract_faus_for_frames(frames)
# print(faus_per_frame)

def main() -> None:
    interim_mp4_list = [
        f"./data/interim/iemocap/Session{i}/" for i in range(1, 6)
    ]

    for video_dir in interim_mp4_list:
        files = list_files_in_folder(video_dir)
        for file in tqdm(files):
            if file.split('.')[-1] != 'mp4':
                continue
            video_path = video_dir + file
            frames = extract_frames(video_path)
            try:
                output_folder = f"./data/processed/{'/'.join(video_dir.split('/')[3:])}"
                os.makedirs(output_folder, exist_ok=True)
                faus_per_frame = extract_faus_for_frames(frames)
                np.savez(f"{output_folder}/{file.split('.')[0]}.npz", **faus_per_frame)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                break

if __name__ == "__main__":
  main()
