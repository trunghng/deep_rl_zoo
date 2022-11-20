import cmds
from agent import DQN


def main():
    args = cmds.get_args()

    agent = DQN(args)
    if args.eval:
        agent.test(args.model_path)
    else:
        agent.train()


if __name__ == '__main__':
    main()
