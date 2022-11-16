import gym
from arguments import get_args
from agent import DQN, DoubleDQN


def main():
    args = get_args()
    agent_name = args.agent
    evaluate = args.eval
    model_path = args.model_path

    del args.agent
    del args.eval
    del args.model_path

    args_ = vars(args)
    
    if agent_name == 'DQN':
        agent = DQN(**args_)
    elif agent_name == 'DoubleDQN':
        agent = DoubleDQN(**args_)

    if evaluate:
        agent.test(model_path)
    else:
        agent.train()


if __name__ == '__main__':
    main()
