import os
import subprocess
import shutil


def create_video_from_intermediate_results(config):
    #
    # change this depending on what you want to accomplish (modify out video name, change fps and trim video)
    #
    dump_path = config['out_videos_path']
    img_pattern = os.path.join(dump_path, '%4d.jpg')
    fps = 5
    first_frame = 0
    number_of_frames_to_process = 30 # len(os.listdir(dump_path))  # default don't trim take process every frame
    out_file_name = 'out.mp4'  # todo: make smarter naming scheme

    ffmpeg = 'ffmpeg.exe'
    if shutil.which(ffmpeg):  # if ffmpeg.exe is in system path
        input_options = ['-r', str(fps), '-i', img_pattern]
        trim_video_command = ['-start_number', str(first_frame), '-vframes', str(number_of_frames_to_process)]
        encoding_options = ['-c:v', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p']
        out_video_path = os.path.join(dump_path, out_file_name)
        subprocess.call([ffmpeg, *input_options, *trim_video_command, *encoding_options, out_video_path])
    else:
        print(f'{ffmpeg} not found in the system path, aborting.')
