from src.h3DQN_FR1.Agent import H_DDQNAgent
from src.h3DQN_FR1.Agent_HL import HL_DDQNAgent
from src.h3DQN_FR1.Agent_LL import LL_DDQNAgent
from src.h3DQN_FR1.ReplayMemory import ReplayMemory
from src.h_CPP.State import H_CPPState
import tqdm


class H_DDQNTrainerParams:
    def __init__(self):
        self.batch_size_h = 64
        self.batch_size_l = 128
        self.num_steps = 1e6
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size_ll = 50000
        self.rm_size_hl = 10000
        self.load_model = ""


class H_DDQNTrainer:
    def __init__(self, params: H_DDQNTrainerParams, agent_ll: LL_DDQNAgent, agent_hl: HL_DDQNAgent):
        self.params = params
        self.replay_memory_ll = ReplayMemory(size=params.rm_size_ll)
        self.replay_memory_hl = ReplayMemory(size=params.rm_size_hl)
        self.l_terminated = False

        self.agent_ll = agent_ll
        self.agent_hl = agent_hl

        self.prefill_bar = None

    def add_experience_ll(self, state, action, reward, next_state):
        self.replay_memory_ll.store((state.get_boolean_map(),
                                  state.get_float_map(),
                                  state.get_scalars(),
                                  action,
                                  reward,
                                  next_state.get_boolean_map_ll(),
                                  next_state.get_float_map(),
                                  next_state.get_scalars_ll(),
                                  next_state.h_terminal))

    def add_experience_hl(self, state, action, reward, next_state):
        self.replay_memory_hl.store((state.get_boolean_map(),
                                  state.get_float_map(),
                                  state.get_scalars(),
                                  action,
                                  reward,
                                  next_state.get_boolean_map(),
                                  next_state.get_float_map(),
                                  next_state.get_scalars_hl(),
                                  next_state.terminal))
        
    def is_h_terminated(self, state: H_CPPState, next_state: H_CPPState):

        return not bool(state.get_remaining_h_target_cells() - next_state.get_remaining_h_target_cells())


    def train_agent(self):
        # train low-level agent
        if self.params.batch_size_l > self.replay_memory_ll.get_size():
            return
        mini_batch = self.replay_memory_ll.sample(self.params.batch_size_l)

        self.agent_ll.train_ll(mini_batch)

        # train high-level agent
        if self.is_h_terminated:
            # print('######### training hl ########')
            if self.params.batch_size_h > self.replay_memory_hl.get_size():
                return
            mini_batch = self.replay_memory_hl.sample(self.params.batch_size_h)

            self.agent_hl.train_hl(mini_batch)

    def should_fill_replay_memory(self):
        target_size = self.replay_memory_ll.get_max_size() * self.params.rm_pre_fill_ratio
        if self.replay_memory_ll.get_size() >= target_size or self.replay_memory_ll.full:
            if self.prefill_bar:
                self.prefill_bar.update(target_size - self.prefill_bar.n)
                self.prefill_bar.close()
            return False

        if self.prefill_bar is None:
            print("Filling replay memory")
            self.prefill_bar = tqdm.tqdm(total=target_size)

        self.prefill_bar.update(self.replay_memory_ll.get_size() - self.prefill_bar.n)

        return True