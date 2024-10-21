import copy

import numpy as np
from utils import numpy_shift, get_valid_advance_actions, get_valid_allocation_actions

class SchedulingEnv:

    def __init__(self,
                 treatment_pattern,
                 decision_epoch,
                 system_dynamic,
                 holding_cost,
                 overtime_cost,
                 duration,
                 regular_capacity,
                 discount_factor,
                 problem_type='allocation'
                 ):
        self.treatment_pattern = treatment_pattern
        self.decision_epoch = decision_epoch
        self.system_dynamic = system_dynamic
        self.holding_cost = holding_cost
        self.overtime_cost = overtime_cost
        self.duration = duration
        self.regular_capacity = regular_capacity
        self.problem_type = problem_type
        self.discount_factor = discount_factor
        if 'allocation' == problem_type:
            self.valid_actions = self.valid_allocation_actions
            self.post_state = self.post_allocation_state
            self.cost_fn = self.allocation_cost

        self.num_sessions, self.num_types = treatment_pattern.shape

    def interpret_state(self, state):
        '''
        (number of bookings, waitlist, future first appointments)
        :param state:
        :return:
        '''
        bookings, waitlist, future_first_appts = state
        return copy.deepcopy(bookings), copy.deepcopy(waitlist), copy.deepcopy(future_first_appts)

    def valid_allocation_actions(self, state, t):
        days_to_go = self.decision_epoch - t
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        if days_to_go <= 0:
            return np.array([waitlist])
        return get_valid_allocation_actions(waitlist)

    def allocation_cost(self, state, action, t):
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        outstanding_treatments = waitlist + future_first_appts.sum(axis=0)
        cost = (self.holding_cost * outstanding_treatments).sum() + self.overtime_cost * np.maximum((bookings[0] + (action * self.treatment_pattern[0]).sum()) * self.duration - self.regular_capacity, 0)
        if t == self.decision_epoch:
            for i in range(1, len(bookings)):
                cost += self.discount_factor ** i * self.overtime_cost * np.maximum(
                    (bookings[i]+(action * self.treatment_pattern[i]).sum()) * self.duration - self.regular_capacity, 0)
        return cost

    def post_allocation_state(self, state, action):
        bookings, waitlist, future_first_appts = self.interpret_state(state)
        next_first_appts = future_first_appts[0] if len(future_first_appts) > 0 else 0
        next_bookings = numpy_shift(bookings + (self.treatment_pattern * (action + next_first_appts)).sum(axis=1), -1)
        next_waitlist = waitlist - action
        return (next_bookings, next_waitlist, future_first_appts[1:])

    def transition_dynamic(self, state, action, t):
        cost = self.cost_fn(state, action, t)
        res = []
        for prob, delta in self.system_dynamic:
            next_bookings, next_waitlist, new_future_first_appts = self.post_state(state, action)
            if t+1 > self.decision_epoch:
                delta = np.zeros(self.num_types, dtype=int)
            next_waitlist += delta
            done = t == self.decision_epoch
            if (next_waitlist.sum() < 0) or (done and next_waitlist.sum() > 0):
                print('time:', t, 'decision_epoch:', self.decision_epoch)
                print(state)
                print(action)
                raise ValueError("Invalid action")
            res.append([prob, (next_bookings, next_waitlist, new_future_first_appts), cost, done])
        return res

    def step(self):
        pass


if __name__ == "__main__":
    treatment_patterns = np.array([[2, 1], [1, 0]]).T
    num_tracking_days, num_types = treatment_patterns.shape
    system_dynamic = [[0.5, np.array([2, 0])], [0.5, np.array([2, 2])]]
    holding_cost = np.array([10, 5])
    overtime_cost = 30
    duration = 1
    regular_capacity = 5
    discount_factor = 0.99

    demand = np.array([6, 6])
    future_appts = np.array([[0, 0], [0, 0]])
    init_state = (np.array([0,0]), demand, future_appts)
    decision_epoch = len(future_appts)

    env = SchedulingEnv(
                 treatment_pattern=treatment_patterns,
                 decision_epoch=decision_epoch,
                 system_dynamic=system_dynamic,
                 holding_cost=holding_cost,
                 overtime_cost=overtime_cost,
                 duration=duration,
                 regular_capacity=regular_capacity,
                 discount_factor=0.99,
                 problem_type='allocation'
    )
    print(env.transition_dynamic(init_state, np.array([6, 6]), 2))
    '''
    action = np.array([[2, 1], [0, 4], [6, 6], [1, 3]])
    print(env.transition_dynamic(init_state,action, 1))
    print(160+0.99*(0.5*(947.45585)+0.5*(1012.20285)))
    '''

