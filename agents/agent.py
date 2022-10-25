import time
from torch.utils.tensorboard import SummaryWriter



# base class for other agent to inherient, interface class
class agent(object):
    def __init__(self,args,logger) -> None:
        self.logger = logger
        time_stamp = time.strftime("%m-%d-%H:%M", time.localtime())
        # if "debug" in args.version:
        #     work_dir = f"./tensorboard_curves/debug/"
        #     args.eval_freq = 1
        #     args.print_freq = 1
        #     args.max_timesteps = 10
        # else:
        work_dir = f"./tensorboard_curves/{args.exp_name}/{args.env}/{args.plot_name}/{args.version}/s_{args.seed}"#/{time_stamp}"

        self.writer = SummaryWriter(work_dir)

    def save_models(self):
        raise NotImplementedError
        self.logger.save_model(self.actor.state_dict(), f"{self.all_steps}_actor")
        self.logger.save_model(self.critic.state_dict(), f"{self.all_steps}_critic")
        self.logger.save_model(self.actor_optimizer.state_dict(), f"{self.all_steps}_actor_optimizer")
        self.logger.save_model(self.q1_optimizer.state_dict(), f"{self.all_steps}_q1_optimizer")
        self.logger.save_model(self.q2_optimizer.state_dict(), f"{self.all_steps}_q2_optimizer")
        self.logger.save_model(self.q3_optimizer.state_dict(), f"{self.all_steps}_q3_optimizer")
        self.logger.save_model(self.q4_optimizer.state_dict(), f"{self.all_steps}_q4_optimizer")
        self.logger.save_model(self.q5_optimizer.state_dict(), f"{self.all_steps}_q5_optimizer")

    def load_models(self):
        raise NotImplementedError
    