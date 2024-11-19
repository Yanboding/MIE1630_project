import numpy as np
import copy

def get_valid_advance_actions(waitlist, number_days):
    if number_days < 1:
        raise ValueError('number_bins must be at least 1')
    def valid_actions_for_one_class(x, y):
        def backtracking(x, y, current_combination):
            if y == 0:
                if x == 0:
                    all_combinations.append(current_combination.copy())
                return

            for i in range(x + 1):
                current_combination.append(i)
                backtracking(x - i, y - 1, current_combination)
                current_combination.pop()

        all_combinations = []
        backtracking(x, y, [])
        return all_combinations

    actions = []

    def backtracking(current, waitlist):
        if len(waitlist) == 0:
            actions.append(copy.deepcopy(current))
            return
        for combination in valid_actions_for_one_class(waitlist[0], number_days):
            current.append(combination)
            backtracking(current, waitlist[1:])
            current.pop()

    backtracking([], waitlist)
    return np.array(actions).transpose((0, 2, 1))

def get_valid_allocation_actions(waitlist):
    actions = []
    def backtracking(current, current_waitlist):
        if len(current_waitlist) == 0:
            actions.append(current.copy())
            return
        for w in range(int(current_waitlist[0] + 1)):
            current.append(w)
            backtracking(current, current_waitlist[1:])
            current.pop()
    backtracking([], waitlist)
    return np.array(actions)

if __name__ == "__main__":
    waitlist = [2, 0]
    days_to_go = 2
    print(get_valid_advance_actions(waitlist, days_to_go))