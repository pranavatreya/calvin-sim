import json
from copy import deepcopy

with open("/nfs/kun2/users/pranav/calvin-sim/initial_states.json") as f:
    initial_states = json.load(f)
with open("/nfs/kun2/users/pranav/calvin-sim/filtered_subtasks.json") as f:
    filtered_subtasks = json.load(f)

# For our reference, let's print all of the values each key can take on
led_values, lightbulb_values, slider_values, drawer_values, red_block_values, blue_block_values, pink_block_values, grasped_values = set(), set(), set(), set(), set(), set(), set(), set()
for initial_state in initial_states:
    led_values.add(initial_state["led"])
    lightbulb_values.add(initial_state["lightbulb"])
    slider_values.add(initial_state["slider"])
    drawer_values.add(initial_state["drawer"])
    red_block_values.add(initial_state["red_block"])
    blue_block_values.add(initial_state["blue_block"])
    pink_block_values.add(initial_state["pink_block"])
    grasped_values.add(initial_state["grasped"])
print("led values:", list(led_values))
print("lightbulb values:", list(lightbulb_values))
print("slider values:", list(slider_values))
print("drawer values:", list(drawer_values))
print("red block values:", list(red_block_values))
print("blue block values:", list(blue_block_values))
print("pink block values:", list(pink_block_values))
print("grasped values:", list(grasped_values))

def to_string(initial_state):
    return str(initial_state["led"]) + "\n" + str(initial_state["lightbulb"]) + "\n" + str(initial_state["slider"]) + "\n" + str(initial_state["drawer"]) + "\n" + str(initial_state["red_block"]) + "\n" + str(initial_state["blue_block"]) + "\n" + str(initial_state["pink_block"]) + "\n" + str(initial_state["grasped"])

# It will be useful to hash all of the initial states
initial_states_set = set()
for initial_state in initial_states:
    initial_states_set.add(to_string(initial_state))

task_state_assignments = {}

num_goal_mismatches = 0

# Let's start with the first task, move_slider_right
start_goal_pairs = []
for initial_state in initial_states:
    # We can use this initial state as a start if the slider is currently to the left
    if initial_state["slider"] != "left":
        continue
    # Next we need to make sure a goal state exists which is exactly identical to this
    # state, except with the slider to the right
    desired_goal = deepcopy(initial_state)
    desired_goal["slider"] = "right"
    if not to_string(desired_goal) in initial_states_set:
        num_goal_mismatches += 1
        continue
    pair = {
        "start" : initial_state,
        "goal" : desired_goal
    }
    start_goal_pairs.append(pair)
task_state_assignments["move_slider_right"] = start_goal_pairs

# Next, turn_on_lightbulb
start_goal_pairs = []
for initial_state in initial_states:
    # We can use this initial state as a start if the lightbulb is currently off
    if initial_state["lightbulb"] != 0:
        continue
    # Next we need to make sure a goal state exists which is exactly identical to this
    # state, except with the lightbulb on
    desired_goal = deepcopy(initial_state)
    desired_goal["lightbulb"] = 1
    if not to_string(desired_goal) in initial_states_set:
        num_goal_mismatches += 1
        continue
    pair = {
        "start" : initial_state,
        "goal" : desired_goal
    }
    start_goal_pairs.append(pair)
task_state_assignments["turn_on_lightbulb"] = start_goal_pairs

# Next, close_drawer
start_goal_pairs = []
for initial_state in initial_states:
    # We can use this initial state as a start if the drawer is currently open
    if initial_state["drawer"] != "open":
        continue
    # Next we need to make sure a goal state exists which is exactly identical to this
    # state, except with the drawer closed
    desired_goal = deepcopy(initial_state)
    desired_goal["drawer"] = "closed"
    if not to_string(desired_goal) in initial_states_set:
        num_goal_mismatches += 1
        continue
    pair = {
        "start" : initial_state,
        "goal" : desired_goal
    }
    start_goal_pairs.append(pair)
task_state_assignments["close_drawer"] = start_goal_pairs

# Next, turn_on_led
start_goal_pairs = []
for initial_state in initial_states:
    # We can use this initial state as a start if the led is currently off
    if initial_state["led"] != 0:
        continue
    # Next we need to make sure a goal state exists which is exactly identical to this
    # state, except with the led on
    desired_goal = deepcopy(initial_state)
    desired_goal["led"] = 1
    if not to_string(desired_goal) in initial_states_set:
        num_goal_mismatches += 1
        continue
    pair = {
        "start" : initial_state,
        "goal" : desired_goal
    }
    start_goal_pairs.append(pair)
task_state_assignments["turn_on_led"] = start_goal_pairs

# Next, turn_off_lightbulb
start_goal_pairs = []
for initial_state in initial_states:
    # We can use this initial state as a start if the lightbulb is currently on
    if initial_state["lightbulb"] != 1:
        continue
    # Next we need to make sure a goal state exists which is exactly identical to this
    # state, except with the lightbulb off
    desired_goal = deepcopy(initial_state)
    desired_goal["lightbulb"] = 0
    if not to_string(desired_goal) in initial_states_set:
        num_goal_mismatches += 1
        continue
    pair = {
        "start" : initial_state,
        "goal" : desired_goal
    }
    start_goal_pairs.append(pair)
task_state_assignments["turn_off_lightbulb"] = start_goal_pairs

# Next, turn_off_led
start_goal_pairs = []
for initial_state in initial_states:
    # We can use this initial state as a start if the led is currently on
    if initial_state["led"] != 1:
        continue
    # Next we need to make sure a goal state exists which is exactly identical to this
    # state, except with the led off
    desired_goal = deepcopy(initial_state)
    desired_goal["led"] = 0
    if not to_string(desired_goal) in initial_states_set:
        num_goal_mismatches += 1
        continue
    pair = {
        "start" : initial_state,
        "goal" : desired_goal
    }
    start_goal_pairs.append(pair)
task_state_assignments["turn_off_led"] = start_goal_pairs

# Next, open_drawer
start_goal_pairs = []
for initial_state in initial_states:
    # We can use this initial state as a start if the drawer is currently closed
    if initial_state["drawer"] != "closed":
        continue
    # Next we need to make sure a goal state exists which is exactly identical to this
    # state, except with the drawer open
    desired_goal = deepcopy(initial_state)
    desired_goal["drawer"] = "open"
    if not to_string(desired_goal) in initial_states_set:
        num_goal_mismatches += 1
        continue
    pair = {
        "start" : initial_state,
        "goal" : desired_goal
    }
    start_goal_pairs.append(pair)
task_state_assignments["open_drawer"] = start_goal_pairs

# Next, move_slider_left
start_goal_pairs = []
for initial_state in initial_states:
    # We can use this initial state as a start if the slider is currently right
    if initial_state["slider"] != "right":
        continue
    # Next we need to make sure a goal state exists which is exactly identical to this
    # state, except with the slider left
    desired_goal = deepcopy(initial_state)
    desired_goal["slider"] = "left"
    if not to_string(desired_goal) in initial_states_set:
        num_goal_mismatches += 1
        continue
    pair = {
        "start" : initial_state,
        "goal" : desired_goal
    }
    start_goal_pairs.append(pair)
task_state_assignments["move_slider_left"] = start_goal_pairs

# Finally write combined json file to disk
with open("/nfs/kun2/users/pranav/calvin-sim/task_state_assignments.json", "w") as f:
    json.dump(task_state_assignments, f, indent=4)

# Print the number of goal mismatches
print("Num goal mismatches:", num_goal_mismatches)