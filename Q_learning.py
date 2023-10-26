#Implements Q learning for a warehouse

import numpy as np


action = ['up','down','right','left']
environment_rows = 11
environment_cols = 11

rewards = np.full((environment_rows,environment_cols),-100)
rewards[0,5] = 100

#q_table = [[[0 for z in range(0,11)] for y in range(0,11)]for x in range(4)]
q_table = np.zeros((environment_rows,environment_cols,len(action)))

travel = {}
travel[1] = [i for i in range(1,10)]
travel[2] = [1,7,9]
travel[3] = [i for i in range(1,8)]
travel[3].append(9)
travel[4] = [3,7]
travel[5] = [i for i in range(11)]
travel[6] = [5]
travel[7] = [i for i in range(1, 10)]
travel[8] = [3, 7]
travel[9] = [i for i in range(11)]

for row in range(1,10):
    for col in travel[row]:
        rewards[row,col] = -1

for row in rewards:
    print(row)

def terminal_state(row_index,col_index):
    if rewards[row_index,col_index] == -1:
        return False
    else:
        return True
    
def get_starting_location():
    current_row_index = np.random.randint(environment_rows)
    current_col_index = np.random.randint(environment_cols)

    while terminal_state(current_row_index,current_col_index):
        current_row_index = np.random.randint(environment_rows)
        current_col_index = np.random.randint(environment_cols)
        
    return current_row_index, current_col_index

def get_next_action(row_index,col_index,epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_table[row_index,col_index])
    else:
        return np.random.randint(4)
    
def get_next_location(current_row_index,current_col_index,action_index):
    new_row_index = current_row_index
    new_col_index = current_col_index

    if action[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif action[action_index] == 'right' and current_col_index < environment_cols-1:
        new_col_index += 1
    elif action[action_index] == 'left' and current_col_index > 0:
        new_col_index -= 1
    elif action[action_index] == 'down' and current_row_index < environment_rows-1:
        new_row_index += 1

    return new_row_index,new_col_index

def shortest_path(start_row, start_col):
    if terminal_state(start_row,start_col):
        return []
    else:
        current_row, current_col = start_row,start_col
        shortest_path = []
        shortest_path.append([current_row,current_col])

        while not terminal_state(current_row,current_col):
            action_index = get_next_action(current_row,current_col,1)
            current_row,current_col = get_next_location(current_row,current_col,action_index)
            shortest_path.append([current_row,current_col])
    return shortest_path


epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9

for i in range(1000):
    row_index,col_index = get_starting_location()

    while not terminal_state(row_index,col_index):
        action_index = get_next_action(row_index,col_index,epsilon)
        old_row_index,old_col_index = row_index,col_index
        row_index,col_index = get_next_location(old_row_index,old_col_index,action_index)
        reward = rewards[row_index,col_index]
        old_q_table = q_table[old_row_index,old_col_index,action_index]
        temporal_difference = reward + (discount_factor * np.max(q_table[row_index, col_index])) - old_q_table
        new_q_table_value = old_q_table + (learning_rate*temporal_difference)
        q_table[old_row_index,old_col_index,action_index] = new_q_table_value
    print('Completed the episode')


print(shortest_path(3,9))
