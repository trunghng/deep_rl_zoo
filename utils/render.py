from gym.wrappers.monitoring import video_recorder

        
def save_video(env, agent, model_path: str, video_path: str):
    '''
    Play video with a saved model

    Parameters
    ----------
    env
    agent
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
    print('Video saved successful!')
    env.close()