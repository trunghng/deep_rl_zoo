from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import base64, io
from os.path import splitext


def play_video(path: str):
    '''
    Play video

    Parameters
    ----------
    path: video path
    '''
    video_ext = splitext(path)[1].split('.')
    try:
        video = io.open(path, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay loop controls
            style="height: 400px;"><source src="data:video/{0};base64,{1}"
            type="video/{0}"/></video>'''.format(video_ext, encoded.decode('ascii'))))
    except FileNotFoundError:
        print('Video not found')
    

        
def play_video_of_model(env, agent, model_path: str, video_path: str):
    '''
    Play video with a saved model

    Parameters
    ----------
    env
    agent
    model_path: model path
    video_path: video path
    '''
    video = video_recorder.VideoRecorder(env, path=video_path)
    agent.load(model_path)
    state = env.reset()
    while True:
        frame = env.render(mode='rgb_array')
        video.capture_frame()
        action = agent.behave(state)
        next_state, reward, terminated, _ = env.step(action)
        if terminated:
            break
        state = next_state
    prin('Video saved successful!')
    env.close()