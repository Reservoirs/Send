import os
import subprocess

# Replace `input_folder` and `output_folder` with your desired directories
input_folder = 'talkbank/wav'
output_folder = 'talkbank/owav'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(input_folder):
    # Check if the file is an MP3
    if file_name.endswith('.mp3'):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.wav')
        
        # Use ffmpeg to convert the file to WAV with a 16kHz sampling rate and a quality level of 8
        subprocess.call(['ffmpeg', '-i', input_path, '-ar', '16000', '-ac', '1', '-q:a', '8', output_path])
