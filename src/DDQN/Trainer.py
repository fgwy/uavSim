from src.DDQN.Agent import DDQNAgent
from src.DDQN.ReplayMemory import ReplayMemory
import tqdm


class DDQNTrainerParams:
    def __init__(self):
        self.batch_size = 128
        self.num_steps = 1e6
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size = 50000
        self.load_model = ""
        self.rm_pre_fill_multiplier = 2
        self.eval = False
        self.eval_greedy = False


class DDQNTrainer:
    def __init__(self, params: DDQNTrainerParams, agent: DDQNAgent):
        self.params = params
        self.replay_memory = ReplayMemory(size=params.rm_size)
        self.agent = agent

        if self.params.load_model != "":
            print("Loading model", self.params.load_model, "for agent")
            self.agent.load_weights(self.params.load_model)

        self.prefill_bar = None

    def add_experience(self, state, action, reward, next_state):
        self.replay_memory.store((state.get_local_map_np(),
                                  state.get_global_map_np(self.agent.params.global_map_scaling, self.agent.params.multimap),
                                  state.get_scalars(),
                                  action,
                                  reward,
                                  next_state.get_local_map_np(),
                                  next_state.get_global_map_np(self.agent.params.global_map_scaling, self.agent.params.multimap),
                                  next_state.get_scalars(),
                                  next_state.terminal))

    def train_agent(self):
        if self.params.batch_size*self.params.rm_pre_fill_multiplier > self.replay_memory.get_size():
            return
        mini_batch = self.replay_memory.sample(self.params.batch_size)

        self.agent.train(mini_batch)

    def should_fill_replay_memory(self):
        target_size = self.replay_memory.get_max_size() * self.params.rm_pre_fill_ratio
        if self.replay_memory.get_size() >= target_size or self.replay_memory.full:
            if self.prefill_bar:
                self.prefill_bar.update(target_size - self.prefill_bar.n)
                self.prefill_bar.close()
            return False

        if self.prefill_bar is None:
            print("Filling replay memory")
            self.prefill_bar = tqdm.tqdm(total=target_size)

        self.prefill_bar.update(self.replay_memory.get_size() - self.prefill_bar.n)

        return True
