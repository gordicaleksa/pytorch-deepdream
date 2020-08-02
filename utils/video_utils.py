import os
import subprocess
import shutil
import re


import cv2 as cv
import imageio


# Return frame names that follow the 6 digit pattern and have .jpg extension
def valid_frames(input_dir):
    def valid_frame_name(str):
        pattern = re.compile(r'[0-9]{6}\.jpg')  # regex, examples it covers: 000000.jpg or 923492.jpg, etc.
        return re.fullmatch(pattern, str) is not None
    candidate_frames = sorted(os.listdir(input_dir))
    valid_frames = list(filter(valid_frame_name, candidate_frames))
    return valid_frames


def create_video_name(config):
    prefix = '' if config['input'].endswith('.mp4') else 'ouroboros_'

    input_name = os.path.basename(config['input']).split('.')[0]
    model_name = config['model'].name
    blend_info = 'blend_none' if config['blend'] is None else f'blend_{config["blend"]}'
    if config['input'].endswith('.mp4'):
        blend_info = ''  # Not used for ouroboros
    infix = f'{input_name}_width_{str(config["img_width"])}_model_{model_name}_{blend_info}'

    suffix = '.mp4'

    video_name = prefix + infix + suffix
    return video_name


def create_video_from_intermediate_results(config, metadata=None):
    # save_and_maybe_display_image uses this same format (it's hardcoded there), not adaptive but does the job
    img_pattern = os.path.join(config['dump_dir'], '%6d.jpg')
    fps = 5 if metadata is None else metadata['fps']
    first_frame = 0
    number_of_frames_to_process = len(valid_frames(config['dump_dir']))  # default - don't trim process every frame
    out_file_name = create_video_name(config)

    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        input_options = ['-r', str(fps), '-i', img_pattern]
        trim_video_command = ['-start_number', str(first_frame), '-vframes', str(number_of_frames_to_process)]
        encoding_options = ['-c:v', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p']
        pad_options = ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']  # libx264 won't work for odd dimensions
        out_video_path = os.path.join(config['out_videos_path'], out_file_name)
        subprocess.call([ffmpeg, *input_options, *trim_video_command, *encoding_options, *pad_options, out_video_path])
        print(f'Saved video to {out_video_path}.')
        return out_video_path
    else:
        raise Exception(f'{ffmpeg} not found in the system path, aborting.')


def dump_frames(video_path, dump_dir):
    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        cap = cv.VideoCapture(video_path)
        fps = int(cap.get(cv.CAP_PROP_FPS))

        input_options = ['-i', video_path]
        extract_options = ['-r', str(fps)]
        out_frame_pattern = os.path.join(dump_dir, 'frame_%6d.jpg')

        subprocess.call([ffmpeg, *input_options, *extract_options, out_frame_pattern])

        print(f'Dumped frames to {dump_dir}.')
        metadata = {'pattern': out_frame_pattern, 'fps': fps}
        return metadata
    else:
        raise Exception(f'{ffmpeg} not found in the system path, aborting.')


def create_gif(frames_dir, out_path):
    assert os.path.splitext(out_path)[1].lower() == '.gif', f'Expected gif got {os.path.splitext(out_path)[1]}.'

    frame_paths = [os.path.join(frames_dir, frame_name) for frame_name in sorted(os.listdir(frames_dir)) if frame_name.endswith('.jpg')]
    images = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimwrite(out_path, images, fps=10)
    print(f'Saved gif to {out_path}.')
