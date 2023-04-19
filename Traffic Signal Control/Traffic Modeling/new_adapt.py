from __future__ import absolute_import
from __future__ import print_function
from select import select
import termios
import os
import sys
import optparse
import subprocess
import random
import time
import cv2
import curses
from keras.optimizers import RMSprop, Adam
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import TensorBoard
import readscreen3
import numpy as np
import datetime
from time import time


def get_options():
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


def generate_routefile_random(episode_length, total_vehicles):
    N_ROADS = 4
    division = constrained_sum_sample_pos(N_ROADS, total_vehicles)
    traffic = []

    for i in np.arange(len(division)):
        traffic.append(division[i] * 0.6)
        traffic.append(division[i] * 0.2)
        traffic.append(division[i] * 0.2)

    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vTypeDistribution id="mixed">
        <vType id="car" vClass="passenger" speedDev="0.2" latAlignment="compact" probability="0.3"/>
        <vType id="moped" vClass="moped" speedDev="0.4" latAlignment="compact" probability="0.7"/>
        </vTypeDistribution>
        <route id="r0" edges="51o 1i 2o 52i"/>
        <route id="r1" edges="51o 1i 4o 54i"/>
        <route id="r2" edges="51o 1i 3o 53i"/>
        <route id="r3" edges="54o 4i 3o 53i"/>
        <route id="r4" edges="54o 4i 1o 51i"/>
        <route id="r5" edges="54o 4i 2o 52i"/>
        <route id="r6" edges="52o 2i 1o 51i"/>
        <route id="r7" edges="52o 2i 4o 54i"/>
        <route id="r8" edges="52o 2i 3o 53i"/>
        <route id="r9" edges="53o 3i 4o 54i"/>
        <route id="r10" edges="53o 3i 1o 51i"/>
        <route id="r11" edges="53o 3i 2o 52i"/>""", file=routes)

        for i in np.arange(len(traffic)):
            print(
                '<flow id="mixed%i" begin="0" end="%i" number="%i" route="r%i" type="mixed" departLane="random" departPosLat="random"/>' % (
                i, episode_length, traffic[i], i), file = routes)

        print("</routes>", file=routes)

    print('TRAFFIC CONFIGURATION - ')
    for i in np.arange(len(traffic)):
        print('Lane %i - %i' % (i+1, traffic[i]))

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>

def generate_routefile():
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
    <vTypeDistribution id="mixed">
        <vType id="car" vClass="passenger" speedDev="0.2" latAlignment="compact" probability="0.3"/>
        <vType id="moped" vClass="moped" speedDev="0.4" latAlignment="compact" probability="0.7"/>
    </vTypeDistribution>
    <route id="r0" edges="51o 1i 2o 52i"/>
    <route id="r1" edges="51o 1i 4o 54i"/>
    <route id="r2" edges="51o 1i 3o 53i"/>
    <route id="r3" edges="54o 4i 3o 53i"/>
    <route id="r4" edges="54o 4i 1o 51i"/>
    <route id="r5" edges="54o 4i 2o 52i"/>
    <route id="r6" edges="52o 2i 1o 51i"/>
    <route id="r7" edges="52o 2i 4o 54i"/>
    <route id="r8" edges="52o 2i 3o 53i"/>
    <route id="r9" edges="53o 3i 4o 54i"/>
    <route id="r10" edges="53o 3i 1o 51i"/>
    <route id="r11" edges="53o 3i 2o 52i"/>
    <flow id="mixed1" begin="0" end="350" number="90" route="r0" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed2" begin="0" end="0" number="0" route="r1" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed3" begin="0" end="0" number="0" route="r2" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed4" begin="0" end="350" number="10" route="r3" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed5" begin="0" end="0" number="0" route="r4" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed6" begin="0" end="0" number="0" route="r5" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed7" begin="0" end="0" number="0" route="r6" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed8" begin="0" end="0" number="0" route="r7" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed9" begin="0" end="0" number="0" route="r8" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed10" begin="0" end="0" number="0" route="r9" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed11" begin="0" end="0" number="0" route="r10" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed12" begin="0" end="0" number="0" route="r11" type="mixed" departLane="random" departPosLat="random"/>
</routes>""", file=routes)
        lastVeh = 0
        vehNr = 0


try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

options = get_options()

# this script has been called from the command line. It will start sumo as a
# server, then connect and run

if options.nogui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')

# first, generate the route file for this simulation

# this is the normal way of using traci. sumo is started as a
# subprocess and then the python script connects and runs


print("TraCI Started")


# State = State_Lengths()
# print(State.get_tails())

# states = State.get_tails


# runner = Runner()
# print(Runner().run)


def getPhaseState(transition_time):
    num_lanes = 4
    num_phases = 4
    phase = traci.trafficlight.getPhase("0")
    phaseState = np.zeros((transition_time,num_lanes,num_phases))
    for i in range(transition_time):
        for j in range(num_lanes):
            phaseState[i][j][phase] = 1
    return phaseState


def getState(transition_time):  # made the order changes
    newState = []
    for _ in range(transition_time):
        traci.simulationStep()

        state = [readscreen3.getLowerQlength() / 80,
             readscreen3.getRightQlength() / 80,
             readscreen3.getUpperQlength() / 80,
             readscreen3.getLeftQlength() / 80
             ]

        newState.insert(0, state)
    # print (state)
    newState = np.array(newState)
    phaseState = getPhaseState(transition_time)
    newState = np.dstack((newState, phaseState))
    newState = np.expand_dims(newState, axis=0)
    return newState


print("here")
import traci


def makeMove(action, transition_time):
    if action == 1:
        traci.trafficlight.setPhase("0", (int(traci.trafficlight.getPhase("0")) + 1) % 4)




    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()

    return getState(transition_time)


def getReward(this_state, this_new_state):
    num_lanes = 4
    qLengths1 = []
    qLengths2 = []
    for i in range(num_lanes):
        qLengths1.append(this_state[0][i][0])
        qLengths2.append(this_new_state[0][i][0])

    qLengths11 = [x + 1 for x in qLengths1]
    qLengths21 = [x + 1 for x in qLengths2]

    q1 = np.prod(qLengths11)
    q2 = np.prod(qLengths21)

    # print("Old State with product : ", q1)
    #
    # print("New State with product : ", q2)
    #
    #
    # if q1 > q2:
    #     this_reward = 1
    # else:
    #     this_reward = -1
    this_reward = q1 - q2

    if this_reward > 0:
        this_reward = 1
    elif this_reward < 0:
        this_reward = -1
    elif q2 > 1:
        this_reward = -1
    else:
        this_reward = 0

    return this_reward


def build_model(transition_time):
    num_hidden_units_cnn = 10
    num_actions = 2
    model = Sequential()
    model.add(Conv2D(num_hidden_units_cnn, kernel_size=(transition_time, 1), strides=1, activation='relu', input_shape=(transition_time, 4,5)))
    # model.add(LSTM(8))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=opt)

    return model


def getWaitingTime(laneID):
    return traci.lane.getWaitingTime(laneID)


num_episode = 121
discount_factor = 0.9
#epsilon = 1
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay_steps = 9000

Average_Q_lengths = []
sum_q_lens = 0
AVG_Q_len_perepisode = []

episode_time = 350
num_vehicles = 250
transition_time = 8
target_update_time = 20
q_estimator_model = load_model('new_model_0709_3_20.h5')
target_estimator_model = load_model('new_model_0709_3_20.h5')
replay_memory_init_size = 35
replay_memory_size = 5000
batch_size = 32
print(q_estimator_model.summary())
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

#generate_routefile_random(episode_time, num_vehicles)
generate_routefile()
traci.start([sumoBinary, "-c", "data/cross.sumocfg",
             "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)

nA = 2

target_estimator_model.set_weights(q_estimator_model.get_weights())

replay_memory = []

for _ in range(replay_memory_init_size):
    '''if traci.simulation.getMinExpectedNumber() <= 0:
        generate_routefile_random(episode_time, num_vehicles)
        traci.load(["--start", "-c", "data/cross.sumocfg",
                    "--tripinfo-output", "tripinfo.xml"]) '''
    state = getState(transition_time)
    action = np.random.choice(np.arange(nA))
    new_state = makeMove(action,transition_time)
    reward = getReward(state,new_state)
    replay_memory.append([state,action,reward,new_state])
    print(len(replay_memory))

total_t = 0
for episode in range(num_episode):
    num_vehicles += 1
    generate_routefile()
    #generate_routefile_random(episode_time, num_vehicles)
    traci.load(["--start", "-c", "data/cross.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)

    state = getState(transition_time)
    counter = 0
    stride = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        print("Episode # ", episode)
        # print("Waiting time on lane 1i_0 = ",getWaitingTime("1i_0"))

        print("Inside episode counter", counter)

        counter += 1
        total_t += 1
        # batch_experience = experience[:batch_history]

        if total_t % target_update_time == 0:
            target_estimator_model.set_weights(q_estimator_model.get_weights())

        q_val = q_estimator_model.predict(state)
        print(q_val)
        # if random.random() < epsilon:
        #     phase = np.random.choice(4)
        #     print("random action chosen",phase)
        # else:
        #     phase = np.argmax(q_val)
        #     print("else action",phase)
        epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
        print("Epsilon -", epsilon)
        policy_s = np.ones(nA) * epsilon / nA

        policy_s[np.argmax(q_val)] = 1 - epsilon + (epsilon / nA)

        action = np.random.choice(np.arange(nA), p=policy_s)

        if np.argmax(q_val) != action:
            print("RANDOM CHOICE TAKEN")
        else:
            print("POLICY FOLLOWED ")

        new_state = makeMove(action, transition_time)
        reward = getReward(state, new_state)

        if len(replay_memory) == replay_memory_size:
            replay_memory.pop(0)

        replay_memory.append([state, action, reward, new_state])

        sum_q_lens += np.average(new_state)

        samples = random.sample(replay_memory, batch_size)
        '''
        states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))

        q_values_next = target_estimator_model.predict(next_states_batch)
        targets_batch = reward_batch + discount_factor * np.amax(
            q_values_next, axis=1)

        states_batch = np.array(states_batch)
        loss = q_estimator_model.update(states_batch, action_batch, targets_batch)
        '''
        # CODE FOR UPDATE REMAINING, REST DONE!
        x_batch, y_batch = [], []
        for inst_state, inst_action, inst_reward, inst_next_state in samples:
            y_target = q_estimator_model.predict(inst_state)
            q_val_next = target_estimator_model.predict(inst_next_state)
            y_target[0][inst_action] = inst_reward + discount_factor * np.amax(
                q_val_next, axis=1
            )
            x_batch.append(inst_state[0])
            y_batch.append(y_target[0])

        q_estimator_model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

        ####

        '''
        oracle = np.zeros((1, nA))
        oracle[:] = q_val[:]
        print(reward)
        oracle[0][action] = (
                    reward + gamma * np.max(model.predict((np.array(experience)).reshape((1, num_history, 5)))))
        print(oracle)
        model.fit((np.array(old_experience)).reshape((1, num_history, 5)), oracle, verbose=1)
        '''
        state = new_state

    AVG_Q_len_perepisode.append(sum_q_lens / 702)
    sum_q_lens = 0
    if episode % 5 == 0:
        q_estimator_model.save('new_model_0709_3_20_adapt_0909_2_{}.h5'.format(episode))



print(AVG_Q_len_perepisode)

# import matplotlib.pyplot as plt
#
# plt.plot([x for x in range(num_episode)],[AVG_Q_len_perepisode], 'ro')
# plt.axis([0, num_episode, 0, 10])
# plt.show()
