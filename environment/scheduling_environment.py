import copy

import numpy as np
import tensorflow as tf
from .utils import numpy_shift, get_valid_advance_actions, get_valid_allocation_actions

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
                 future_first_appts=None,
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
        if future_first_appts == None:
            future_first_appts = np.array([[0]*self.num_types for _ in range(decision_epoch)])
            self.future_first_appts = future_first_appts
            self.future_first_appts_copy = copy.deepcopy(future_first_appts)
        self.probabilities = [item[0] for item in system_dynamic]
        self.arrivals = [item[1] for item in system_dynamic]

    def interpret_state(self, state):
        '''
        (number of bookings, waitlist, future first appointments)
        :param state:
        :return:
        '''
        bookings, waitlist = state
        return copy.deepcopy(bookings), copy.deepcopy(waitlist)

    def valid_allocation_actions(self, state, t):
        days_to_go = self.decision_epoch - t
        bookings, waitlist = self.interpret_state(state)
        if days_to_go <= 0:
            return np.array([waitlist])
        return get_valid_allocation_actions(waitlist)

    def allocation_cost(self, state, action, t):
        bookings, waitlist = self.interpret_state(state)
        outstanding_treatments = waitlist + np.sum(self.future_first_appts, axis=0) 
        cost = tf.reduce_sum(self.holding_cost * outstanding_treatments)
        overtime = tf.maximum((bookings[0] + np.sum(action * self.treatment_pattern[0])) * self.duration - self.regular_capacity, 0)
        cost += self.overtime_cost * overtime
        if t == self.decision_epoch:
            for i in range(1, len(bookings)):
                discounted_overtime = self.discount_factor ** i * self.overtime_cost * tf.maximum(
                    (bookings[i] + np.sum(action * self.treatment_pattern[i])) * self.duration - self.regular_capacity, 0)
                cost += discounted_overtime
        print(self.holding_cost.shape)
        print(outstanding_treatments.shape)
        return cost

    def post_allocation_state(self, state, action):
        bookings, waitlist = self.interpret_state(state)
        next_first_appts = self.future_first_appts[0] if len(self.future_first_appts) > 0 else 0
        next_bookings = numpy_shift(bookings + tf.reduce_sum(self.treatment_pattern * (action + next_first_appts), axis=1), -1)  # Using TensorFlow (tf.reduce_sum)
        next_waitlist = waitlist - action
        self.future_first_appts = self.future_first_appts[1:]
        return (next_bookings, next_waitlist)

    def transition_dynamic(self, state, action, t):
        cost = self.cost_fn(state, action, t)
        res = []
        for prob, delta in self.system_dynamic:
            next_bookings, next_waitlist = self.post_state(state, action)
            if t+1 > self.decision_epoch:
                delta = np.zeros(self.num_types, dtype=int)
            next_waitlist += delta
            done = t == self.decision_epoch
            if (next_waitlist.sum() < 0) or (done and next_waitlist.sum() > 0):
                print('time:', t, 'decision_epoch:', self.decision_epoch)
                print(state)
                print(action)
                raise ValueError("Invalid action")
            res.append([prob, (next_bookings, next_waitlist), cost, done])
        return res

    def reset(self):
        np.random.seed(42)
        self.t = 1
        booked_slots = np.array([0 for _ in range(self.num_sessions)])
        demand = np.array([0 for _ in range(self.num_types)])
        self.future_first_appts = copy.deepcopy(self.future_first_appts_copy)
        self.state = (booked_slots, demand)
        return np.concatenate(self.interpret_state(self.state)), {"future_appointment": copy.deepcopy(self.future_first_appts)}

    def step(self, action):
        cost = self.cost_fn(self.state, action, self.t)
        next_bookings, next_waitlist = self.post_state(self.state, action)
        selected_index = np.random.choice(len(self.arrivals), p=self.probabilities)
        delta = self.arrivals[selected_index]
        next_waitlist += delta
        self.t += 1
        self.state = (next_bookings, next_waitlist)
        done = self.t == self.decision_epoch
        # state, cost, is_done, is_truncate, info
        return np.concatenate(self.interpret_state(self.state)), cost, done, False, {"future_appointment": copy.deepcopy(self.future_first_appts)}
