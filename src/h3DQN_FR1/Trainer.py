from src.h3DQN_FR1.Agent import H_DDQNAgent
from src.h3DQN_FR1.Agent_HL import HL_DDQNAgent
from src.h3DQN_FR1.Agent_LL import LL_DDQNAgent
from src.h3DQN_FR1.ReplayMemory import ReplayMemory
from src.h_CPP.State import H_CPPState
import tqdm


class H_DDQNTrainerParams:
    def __init__(self):
        self.batch_size_h = 128
        self.batch_size_l = 128
        self.num_steps = 1e6
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 100
        self.rm_size_ll = 50000
        self.rm_size_hl = 50000
        self.load_model = ""
        self.use_astar = False
        self.rm_pre_fill_multiplier_ll = 10
        self.rm_pre_fill_multiplier_hl = 5



class H_DDQNTrainer:
    def __init__(self, params: H_DDQNTrainerParams, agent_ll: LL_DDQNAgent, agent_hl: HL_DDQNAgent):
        self.params = params
        self.replay_memory_ll = ReplayMemory(size=params.rm_size_ll)
        self.replay_memory_hl = ReplayMemory(size=params.rm_size_hl)
        self.smdp_terminated = False

        self.agent_ll = agent_ll
        self.agent_hl = agent_hl

        self.prefill_bar = None
        self.i = 0
        self.n = 0

    def add_experience_ll(self, state, action, reward, next_state):
        # print(f'action: {action}')
        self.replay_memory_ll.store((state.get_local_map_ll_np(),
                                  # state.get_global_map_ll(self.agent_hl.params.global_map_scaling),
                                  state.get_scalars_ll(),
                                  action,
                                  reward,
                                  next_state.get_local_map_ll_np(),
                                  # next_state.get_global_map_ll(self.agent_hl.params.global_map_scaling),
                                  next_state.get_scalars_ll(),
                                  next_state.h_terminal))

    def add_experience_hl(self, state, action, reward, next_state):
        # print(f'state get local map: {state.get_local_map().shape, state.get_local_map()}')
        self.replay_memory_hl.store((state.get_local_map_np(),
                                  state.get_global_map_np(self.agent_hl.params.global_map_scaling, self.agent_hl.params.multimap),
                                  state.get_scalars_hl(),
                                  action,
                                  reward,
                                  next_state.get_local_map_np(),
                                  next_state.get_global_map_np(self.agent_hl.params.global_map_scaling, self.agent_hl.params.multimap),
                                  next_state.get_scalars_hl(),
                                  next_state.terminal))
        
    def is_h_terminated(self, state: H_CPPState, next_state: H_CPPState):
        # TODO: change check for termination
        # return not bool(state.get_remaining_h_target_cells() - next_state.get_remaining_h_target_cells())
        return next_state.is_terminal_h()


    def train_h(self):

        if self.params.batch_size_h*self.params.rm_pre_fill_multiplier_hl > self.replay_memory_hl.get_size():
            # print("Filling replay memory to get enough data")
            return
        mini_batch = self.replay_memory_hl.sample(self.params.batch_size_h)
        if self.n == 0:
            self.n += 1
            print('############# training h #################################')
        self.agent_hl.train_hl(mini_batch)

    def train_l(self):
        if self.params.batch_size_l*self.params.rm_pre_fill_multiplier_ll > self.replay_memory_ll.get_size():
            # print("Filling replay memory to get enough data")
            return
        mini_batch = self.replay_memory_ll.sample(self.params.batch_size_l)
        if self.i == 0:
            self.i += 1
            print('########### training l ######################')
        self.agent_ll.train_ll(mini_batch)

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