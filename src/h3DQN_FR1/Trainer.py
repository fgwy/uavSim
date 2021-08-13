from src.h3DQN_FR1.Agent import H_DDQNAgent
from src.h3DQN_FR1.ReplayMemory import ReplayMemory
from src.h_CPP.State import h_CPPState
import tqdm


class H_DDQNTrainerParams:
    def __init__(self):
        self.batch_size = 128
        self.num_steps = 1e6
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size_l = 50000
        self.rm_size_h = 50000
        self.load_model = ""


class H_DDQNTrainer:
    def __init__(self, params: H_DDQNTrainerParams, agent: H_DDQNAgent):
        self.params = params
        self.replay_memory_l = ReplayMemory(size=params.rm_size_l)
        self.replay_memory_h = ReplayMemory(size=params.rm_size_h)
        self.agent = agent
        self.l_terminated = False

        if self.params.load_model != "":
            print("Loading model", self.params.load_model, "for agent")
            self.agent.load_weights_l(self.params.load_model)
            self.agent.load_weights_h(self.params.load_model)

        self.prefill_bar = None

    def add_experience_l(self, state, action, reward, next_state):
        self.replay_memory_l.store((state.get_boolean_map(),
                                  state.get_float_map(),
                                  state.get_scalars(),
                                  action,
                                  reward,
                                  next_state.get_boolean_map(),
                                  next_state.get_float_map(),
                                  next_state.get_scalars(),
                                  next_state.terminal))

    def add_experience_h(self, state, action, reward, next_state):
        self.replay_memory_h.store((state.get_boolean_map(),
                                  state.get_float_map(),
                                  state.get_scalars(),
                                  action,
                                  reward,
                                  next_state.get_boolean_map(),
                                  next_state.get_float_map(),
                                  next_state.get_scalars(),
                                  next_state.terminal))
        
    def is_terminated(self, state: h_CPPState, next_state: h_CPPState):

        return not bool(state.get_remaining_h_target_cells() - next_state.get_remaining_h_target_cells())


    def train_agent(self):
        # train low-level agent
        if self.params.batch_size > self.replay_memory_l.get_size():
            return
        mini_batch = self.replay_memory_l.sample(self.params.batch_size)

        self.agent.train_l(mini_batch)

        # train high-level agent
        if self.is_terminated:
            if self.params.batch_size > self.replay_memory_h.get_size():
                return
            mini_batch = self.replay_memory_h.sample(self.params.batch_size)

            self.agent.train_h(mini_batch)

    def should_fill_replay_memory(self):
        target_size = self.replay_memory_l.get_max_size() * self.params.rm_pre_fill_ratio
        if self.replay_memory_l.get_size() >= target_size or self.replay_memory_l.full:
            if self.prefill_bar:
                self.prefill_bar.update(target_size - self.prefill_bar.n)
                self.prefill_bar.close()
            return False

        if self.prefill_bar is None:
            print("Filling replay memory")
            self.prefill_bar = tqdm.tqdm(total=target_size)

        self.prefill_bar.update(self.replay_memory_l.get_size() - self.prefill_bar.n)

        return True