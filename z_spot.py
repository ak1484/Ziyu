import os
import cv2
from zipfile import ZipFile
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from scenedetect import FrameTimecode
import pandas as pd
import tempfile
from scenedetect import VideoManager, SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors.content_detector import ContentDetector
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import re
import base64
import openai
import pyperclip

from config import gpt_api_key, gpt_model_id

def srt_to_df(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s((?:(?!\d{2}:\d{2}:\d{2},\d{3}).)*?)\n\n'
        matches = re.findall(pattern, content, re.DOTALL)

        df = pd.DataFrame(matches, columns=['start_time', 'end_time', 'dialogue'])
        df['dialogue'] = df['dialogue'].str.replace('\n', ' ')  # replace newlines in dialogue with spaces

        return df

def to_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download csv file</a>'
    return href

def to_tsv_download_link(df, filename):
    tsv = df.to_csv(index=False, sep='\t')
    b64 = base64.b64encode(tsv.encode()).decode()
    href = f'<a href="data:file/tsv;base64,{b64}" download="{filename}">Download tsv file</a>'
    return href


def find_scenes(video_path):
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()
    scene_df = pd.DataFrame(columns=["Scene Number", "Start Time", "Start Frame", "End Time", "End Frame", "Shot Duration"])
    print('Done')
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_timecode()
        start_frame = scene[0].get_frames()
        #fps = scene[0].get_fps()

        end_time = scene[1].get_timecode()
        end_frame = scene[1].get_frames()

        # Add a new row to the dataframe
        scene_df.loc[i] = [i+1, start_time, end_time, start_frame, end_frame, str(pd.to_datetime(end_time) - pd.to_datetime(start_time))]


    return scene_list, scene_df

def is_image_similar(imageA, imageB, threshold=0.5):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    score = ssim(grayA, grayB)
    return score > threshold


def save_shots(s_list, filepath, folder):
    progress = st.progress(0)

    scene_nos = []
    image_paths = []
    # image_urls = [] # If you need to store the URLs for the images uploaded to imgur, uncomment this line.
    
    total_iterations = len(s_list)

    with ZipFile(f"{folder}.zip", "w") as zipf:
        for i, scene in enumerate(s_list, start=1):
            
            progress_value = min(i / total_iterations, 1)
            progress.progress(progress_value)
            
            scene_nos.append(i)
            start_time = scene[0]
            end_time = scene[1]
            framerate = scene[0].get_framerate()

            duration = end_time.get_frames() - start_time.get_frames()
            duration_seconds = duration / framerate

            if duration_seconds > 35:
                num_images = 6
            elif duration_seconds > 25:
                num_images = 5
            elif duration_seconds > 15:
                num_images = 4
            else:
                num_images = 3

            frame_jump = duration // (num_images-1)
            i_paths = []
            # i_urls = [] # If you need to store the URLs for the images uploaded to imgur, uncomment this line.

            previous_images = []
            for j in range(num_images):

                video_manager = VideoManager([filepath])
                frame_time = FrameTimecode(start_time.get_frames() + j * (frame_jump - 1) + 1, video_manager.get_framerate())
                video_manager.set_duration(start_time=frame_time, end_time=frame_time)

                video_manager.start()
                v_tup = video_manager.retrieve()
                ret_val = v_tup[0]
                frame_im = v_tup[1]
                video_manager.release()

                if type(frame_im) != type(None):
                    is_similar = any(is_image_similar(frame_im, prev_im) for prev_im in previous_images)

                    if not is_similar:
                        image_name = 'Scene-{}-{}.jpg'.format(i, frame_time.frame_num)
                        i_paths.append(image_name)
                        cv2.imwrite(image_name, frame_im)

                        # image = client.upload_from_path(image_name, config=None, anon=True) # If you need to upload the images to imgur, uncomment these lines.
                        # i_urls.append(image['link']) # If you need to store the URLs for the images uploaded to imgur, uncomment this line.

                        previous_images.append(frame_im)

                        zipf.write(image_name)
                        os.remove(image_name)  # remove the image file after adding it to the zip file

            image_paths.append(', '.join(i_paths))
            # image_urls.append(', '.join(i_urls)) # If you need to store the URLs for the images uploaded to imgur, uncomment this line.

    progress.progress(100)
    
    snapshots = pd.DataFrame({'Shot Number': scene_nos, 'Image Paths': image_paths})
    
    return snapshots

def make_final_df(transcript, scenes, snaps):

    transcript['start_time'] = pd.to_datetime(transcript['start_time'])
    transcript['end_time'] = pd.to_datetime(transcript['end_time'])
    scenes['Start Time'] = pd.to_datetime(scenes['Start Time'])
    scenes['End Time'] = pd.to_datetime(scenes['End Time'])

    # Assign a constant key to each dataframe and merge
    transcript['key'] = 0
    scenes['key'] = 0
    merged_df = pd.merge(scenes, transcript, on='key')

    # Filter rows
    semifinal_df = merged_df[((merged_df['Start Time'] <= merged_df['start_time']) & 
                          (merged_df['End Time'] >= merged_df['end_time'])) |
                         ((merged_df['Start Time'] <= merged_df['end_time']) & 
                          (merged_df['Start Time'] >= merged_df['start_time']))]

    # Drop the key column
    semifinal_df = semifinal_df.drop('key', axis=1)

    final_df = pd.merge(scenes, semifinal_df, on = "Scene Number", how = 'left')
    final_df = final_df[['Scene Number', 'Start Time_x', 'End Time_x', 'Start Frame_x',
                         'End Frame_x', 'Shot Duration_x', 'start_time','end_time', 'dialogue']]

    final_df.columns = ['Shot Number', 'Shot Start Time', 'Shot End Time', 'Start Frame Number',
                        'End Frame Number', 'Duration of the Scene', 'Dialogue Start Time', 'Dialogue End Time', 'Dialogue']

    final_df['Shot Start Time'] = final_df['Shot Start Time'].dt.strftime('%H:%M:%S.%f')
    final_df['Shot End Time'] = final_df['Shot End Time'].dt.strftime('%H:%M:%S.%f')
    final_df['Dialogue Start Time'] = final_df['Dialogue Start Time'].dt.strftime('%H:%M:%S.%f')
    final_df['Dialogue End Time'] = final_df['Dialogue End Time'].dt.strftime('%H:%M:%S.%f')

    final_df = pd.merge(final_df, snaps, on = 'Shot Number')
    # final_df.to_csv('{}_final_df.csv'.format(folder_name))

    return final_df


def ChatGPT_conversation(conversation, model_id = gpt_model_id):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation
    )
    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation

def brand_relevance(transcript, API_KEY = gpt_api_key, model_id = gpt_model_id):
    transcript.to_clipboard(index=False)
    openai.api_key = API_KEY

    conversation=[]

    prompt = """Following is the transcript of one of the episodes of a tv series. You have to identify the different contexts of the several parts of this transcript, each more than at least 3-4 dialogues long, and identify what kind of brand categories can be displayed there. Tell me the starting time and the ending time of that part, the context during that time frame, the relevant brand categories, why you think so, and a relevance score from 1-10. Give me at least 10 such instances.: {}""".format(pyperclip.paste())

    conversation.append({'role': 'user', 'content': prompt})
    conversation = ChatGPT_conversation(conversation)
    
    #return conversation[-1]['role'].strip(), conversation[-1]['content'].strip()
    
    st.write('\n{0}: {1}\n'.format(conversation[-1]['role'].strip(), conversation[-1]['content'].strip()))

def button_count(key):
    ss[key] += 1

def main():
    st.title('Z-Spot Brand Relevance')

    st.write('Please upload your video and subtitle files:')

    video_file = st.file_uploader('Upload your .mp4 file', type=['mp4'])
    subtitle_file = st.file_uploader('Upload your .srt file', type=['srt'])
    
    if 'b1_count' not in ss:
        ss['b1_count'] = 0
    if 'b2_count' not in ss:
        ss['b2_count'] = 0
    
    st.button('Start the Process', key = 'b1', on_click = button_count, args = ['b1_count', ])
    button1 = bool(ss.b1_count > 0)
    
    if button1:
        if video_file is not None and subtitle_file is not None:
            with tempfile.TemporaryDirectory() as tempdir:
                subtitle_path = os.path.join(tempdir, "temp.srt")
                video_path = os.path.join(tempdir, "temp.mp4")
                with open(subtitle_path, 'wb') as f:
                    f.write(subtitle_file.getbuffer())
                with open(video_path, 'wb') as f:
                    f.write(video_file.getbuffer())
                    

                transcript_df = srt_to_df(subtitle_path)
                st.dataframe(transcript_df)
                csv_filename = video_file.name.split('.')[0] + '_transcript.csv'
                tsv_filename = video_file.name.split('.')[0] + '_transcript.tsv'
                st.markdown(to_csv_download_link(transcript_df, csv_filename), unsafe_allow_html=True)
                st.markdown(to_tsv_download_link(transcript_df, tsv_filename), unsafe_allow_html=True)
                
                placeholder = st.empty()
                placeholder.text('Identifying Shots in the video (Might take a while)...')
                scene_list, scene_df = find_scenes(video_path)
                placeholder.empty()
                st.dataframe(scene_df)
                csv_filename = video_file.name.split('.')[0] + '_scenes.csv'
                st.markdown(to_csv_download_link(scene_df, csv_filename), unsafe_allow_html=True)
                
                st.write('Saving Relevant Snapshots from the identified shots')
                snapshots = save_shots(scene_list, video_path, video_file.name.split('.')[0])
                st.write("Images saved in your working directory in a compressed folder named {}".format(video_file.name.split('.')[0]))
                
                final_df = make_final_df(transcript_df, scene_df, snapshots)
                st.dataframe(final_df)
                csv_filename = video_file.name.split('.')[0] + '_final_df.csv'
                st.markdown(to_csv_download_link(final_df, csv_filename), unsafe_allow_html=True)
                
                st.button('Generate Brand Relevance', key = 'b2', on_click = button_count, args = ['b2_count',])
                button2 = bool(ss.b2_count > 0)    
        
        
        else:
            st.write('Please make sure both video and subtitle files are uploaded.')
        
        if button2:
            brand_relevance(transcript_df)
                

if __name__ == "__main__":
    main()