from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import optparse
import random
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import numpy as np
import traci
import matplotlib.pyplot as plt


def get_options():
    """
    Get command line options.
    """
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


try:
    # Add SUMO tools directory to the system path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools"))

    # Add SUMO_HOME tools directory to the system path
    sys.path.append(
        os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools"))

    # Import the checkBinary module from sumolib
    from sumolib import checkBinary  # noqa
except ImportError:
    # If ImportError occurs, exit with a message asking to declare the 'SUMO_HOME' environment variable
    sys.exit(
        "Please declare the 'SUMO_HOME' environment variable as the root directory of your SUMO installation (it "
        "should contain folders 'bin', 'tools', and 'docs')")

options = get_options()

if options.nogui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')

print("TraCI Started")


def getPhaseState(transition_time):
    """
    Get the state of traffic light phases over a transition period.

    Parameters:
    - transition_time (int): The duration of the transition period.

    Returns:
    - np.ndarray: A 3D NumPy array representing the phase state with shape (transition_time, num_lanes, num_phases).
    """
    num_lanes = 4
    num_phases = 4
    phase = traci.trafficlight.getPhase("0")

    # Initialize a NumPy array filled with zeros
    phaseState = np.zeros((transition_time, num_lanes, num_phases))

    # Set the phase state for each time step in the transition period
    for i in range(transition_time):
        for j in range(num_lanes):
            phaseState[i][j][phase] = 1

    return phaseState


def getState(transition_time):
    """
    Get the current state of the traffic environment over a transition period.

    Parameters:
    - transition_time (int): The duration of the transition period.

    Returns:
    - np.ndarray: A 3D NumPy array representing the state with shape (1, transition_time, num_features),
                  where num_features includes traffic counts and phase states.
    """
    new_state = []

    for _ in range(transition_time):
        traci.simulationStep()

        # Count of vehicles in different regions
        left_count = 0
        right_count = 0
        top_count = 0
        bottom_count = 0

        vehicle_list = traci.vehicle.getIDList()

        for vehicle_id in vehicle_list:
            x, y = traci.vehicle.getPosition(vehicle_id)

            if 60 < x < 110 and 120 < y < 130:
                left_count += 1
            elif 110 < x < 120 and 60 < y < 110:
                bottom_count += 1
            elif 130 < x < 180 and 110 < y < 120:
                right_count += 1
            elif 120 < x < 130 and 130 < y < 180:
                top_count += 1

        # Print traffic counts
        print("Left : ", left_count)
        print("Right : ", right_count)
        print("Top : ", top_count)
        print("Bottom : ", bottom_count)

        # Normalize traffic counts
        state = [bottom_count / 40,
                 right_count / 40,
                 top_count / 40,
                 left_count / 40
                 ]

        # Insert the state at the beginning of the list
        new_state.insert(0, state)

    # Convert the list to a NumPy array
    new_state = np.array(new_state)

    # Get the phase state using the getPhaseState function
    phase_state = getPhaseState(transition_time)

    # Stack the traffic counts and phase state along the third dimension
    new_state = np.dstack((new_state, phase_state))

    # Add an extra dimension to the array
    new_state = np.expand_dims(new_state, axis=0)

    return new_state


def makeMove(action, transition_time):
    """
    Perform a traffic light phase change based on the action and return the new state.

    Parameters:
    - action (int): The action representing the traffic light phase change.
    - transition_time (int): The duration of the transition period.

    Returns:
    - np.ndarray: A 3D NumPy array representing the new state after the phase change.
    """
    if action == 1:
        current_phase = int(traci.trafficlight.getPhase("0"))
        traci.trafficlight.setPhase("0", (current_phase + 1) % 4)

    # Return the new state after the phase change
    return getState(transition_time)


def getReward(this_state, this_new_state):
    """
    Calculate the reward based on the difference in queue lengths between two states.

    Parameters:
    - this_state (np.ndarray): The current state.
    - this_new_state (np.ndarray): The new state.

    Returns:
    - int: The calculated reward (-1, 0, or 1).
    """
    num_lanes = 4

    # Extract queue lengths from the states
    qLengths1 = [this_state[0][0][i][0] + 1 for i in range(num_lanes)]
    qLengths2 = [this_new_state[0][0][i][0] + 1 for i in range(num_lanes)]

    # Calculate the product of queue lengths
    q1 = np.prod(qLengths1)
    q2 = np.prod(qLengths2)

    # Calculate the reward based on the difference in queue lengths
    this_reward = q1 - q2

    # Adjust the reward based on certain conditions
    if this_reward > 0:
        this_reward = 1
    elif this_reward < 0:
        this_reward = -1
    elif q2 > 1:
        this_reward = -1
    else:
        this_reward = 0

    return this_reward


def getRewardAbsolute(this_state, this_new_state):
    """
    Calculate the absolute reward based on the cubic difference in queue lengths between two states.

    Parameters:
    - this_state (np.ndarray): The current state.
    - this_new_state (np.ndarray): The new state.

    Returns:
    - float: The calculated absolute reward.
    """
    num_lanes = 4

    # Extract queue lengths from the states
    qLengths1 = [this_state[0][0][i][0] + 1 for i in range(num_lanes)]
    qLengths2 = [this_new_state[0][0][i][0] + 1 for i in range(num_lanes)]

    # Calculate the product of queue lengths
    q1 = np.prod(qLengths1)
    q2 = np.prod(qLengths2)

    # Calculate the reward based on the cubic difference in queue lengths
    this_reward = q1 - q2
    this_reward_cubic = this_reward ** 3

    return this_reward_cubic


def build_model(transition_time):
    """
    Build a convolutional neural network model for the traffic signal control.

    Parameters:
    - transition_time (int): The transition time for the traffic signal.

    Returns:
    - keras.models.Sequential: The constructed neural network model.
    """
    num_hidden_units_cnn = 10
    num_actions = 2

    # Create a Sequential model
    model = Sequential()

    # Add a 1D convolutional layer
    model.add(Conv2D(num_hidden_units_cnn, kernel_size=(transition_time, 1), strides=1, activation='relu',
                     input_shape=(transition_time, 4, 5)))

    # Flatten the output
    model.add(Flatten())

    # Add a dense layer with ReLU activation
    model.add(Dense(20, activation='relu'))

    # Add the output layer with linear activation for Q-values
    model.add(Dense(num_actions, activation='linear'))

    # Compile the model with Mean Squared Error loss and RMSprop optimizer
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=opt)

    return model


def getWaitingTime(laneID):
    """
    Get the waiting time for vehicles in a specific lane.

    Parameters:
    - laneID (str): The ID of the lane.

    Returns:
    - float: The waiting time in seconds for vehicles in the specified lane.
    """
    return traci.lane.getWaitingTime(laneID)


num_episode = 16
discount_factor = 0.9
epsilon_start = 1
epsilon_end = 0.4
epsilon_decay_steps = 3000
Average_Q_lengths = []
params_dict = []
sum_q_lens = 0
AVG_Q_len_perepisode = []
transition_time = 8
target_update_time = 20
q_estimator_model = build_model(transition_time)
target_estimator_model = build_model(transition_time)
replay_memory_init_size = 350
replay_memory_size = 8000
batch_size = 32
print(q_estimator_model.summary())
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

traci.start([sumoBinary, "-c", "data/cross.sumocfg",
             "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)

nA = 2

target_estimator_model.set_weights(q_estimator_model.get_weights())

replay_memory = []

for _ in range(replay_memory_init_size):
    if traci.simulation.getMinExpectedNumber() <= 0:
        traci.load(["--start", "-c", "data/cross.sumocfg",
                    "--tripinfo-output", "tripinfo.xml"])
    state = getState(transition_time)
    action = np.random.choice(np.arange(nA))
    new_state = makeMove(action, transition_time)
    reward = getRewardAbsolute(state, new_state)
    replay_memory.append([state, action, reward, new_state])
    print(len(replay_memory))

total_t = 0
for episode in range(num_episode):
    traci.load(["--start", "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)

    state = getState(transition_time)
    counter = 0

    delay_data_avg = []
    delay_data_min = []
    delay_data_max = []
    delay_data_time = []

    while traci.simulation.getMinExpectedNumber() > 0:
        counter += 1
        total_t += 1

        if total_t % target_update_time == 0:
            target_estimator_model.set_weights(q_estimator_model.get_weights())

        q_val = q_estimator_model.predict(state)
        epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
        policy_s = np.ones(nA) * epsilon / nA
        policy_s[np.argmax(q_val)] = 1 - epsilon + (epsilon / nA)
        action = np.random.choice(np.arange(nA), p=policy_s)

        same_action_count = sum(1 for temp in reversed(replay_memory) if temp[1] == 0)
        if same_action_count == 20:
            action = 1
            print("SAME ACTION PENALTY")

        if np.argmax(q_val) != action:
            print("RANDOM CHOICE TAKEN")
        else:
            print("POLICY FOLLOWED ")

        new_state = makeMove(action, transition_time)
        reward = getRewardAbsolute(state, new_state)

        vehicleList = traci.vehicle.getIDList()
        num_vehicles = len(vehicleList)
        if num_vehicles:
            avg, max_val, mini = 0, 0, 100
            for id in vehicleList:
                time = traci.vehicle.getAccumulatedWaitingTime(id)
                max_val = max(max_val, time)
                mini = min(mini, time)
                avg += time
            avg /= num_vehicles
            delay_data_avg.append(avg)
            delay_data_max.append(max_val)
            delay_data_min.append(mini)
            delay_data_time.append(traci.simulation.getCurrentTime() / 1000)

        if len(replay_memory) == replay_memory_size:
            replay_memory.pop(0)

        replay_memory.append([state, action, reward, new_state])

        print("Memory Length:", len(replay_memory))

        sum_q_lens += np.average(new_state)

        samples = random.sample(replay_memory, batch_size)
        x_batch, y_batch = [], []
        for inst_state, inst_action, inst_reward, inst_next_state in samples:
            y_target = q_estimator_model.predict(inst_state)
            q_val_next = target_estimator_model.predict(inst_next_state)
            y_target[0][inst_action] = inst_reward + discount_factor * np.amax(q_val_next, axis=1)
            x_batch.append(inst_state[0])
            y_batch.append(y_target[0])

        q_estimator_model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

        state = new_state

    AVG_Q_len_perepisode.append(sum_q_lens / 702)
    sum_q_lens = 0

    q_estimator_model.save(
        'models/single intersection models/tradeoff_models_absreward_cubic/model_{}.h5'.format(episode))

print(AVG_Q_len_perepisode)

plt.plot([x for x in range(num_episode)], [AVG_Q_len_perepisode], 'ro')
plt.axis([0, num_episode, 0, 10])
plt.show()
