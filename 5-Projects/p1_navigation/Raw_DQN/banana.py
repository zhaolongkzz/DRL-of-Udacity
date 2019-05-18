import numpy as np
from collections import deque

def banana(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    scores_window = deque(maxlen=100)                   # last 100 scores
    scores = []                                         # initialize the score
    eps = eps_start                                     # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_banana = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_banana.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)                   # select an action
            env_banana = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_banana.vector_observations[0]   # get the next state
            reward = env_banana.rewards[0]                   # get the reward
            done = env_banana.local_done[0]                  # see if episode has finished
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state                             # roll over the state to next time step
            score += reward                                # update the score
            if done:                                       # exit loop if episode finished
                break
        scores_window.append(score)                    # save most recent score
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)              # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'dqn_checkpoint.pth')
            break
            
    return scores