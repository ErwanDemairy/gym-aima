import gymnasium as gym
import gym_aima, numpy as np
def value_iteration(P, gamma=0.99, theta = 1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, new_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[new_state] * (not done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}
    return V, pi
    
def print_policy(pi, P):
    arrs = {k:v for k,v in enumerate(('<', 'v', '>', '^'))}
    for key, value in pi.items():
        print("| ", end="")
        if P[key][0][0][0] == 1.0:
            print("    ", end=" ")
        else:
            print(str(key).zfill(2), arrs[value], end=" ")
        if (key + 1) % 4 == 0: print("|")
        
# reproduce Russell & Norvig
env = gym.make('RussellNorvigGridworld-v0')
V_best_v, pi_best_v = value_iteration(env.P, gamma=1.0)
print(f"V_best_v = {V_best_v}")

print(f"pi_best_v() = {pi_best_v}")
print_policy(pi_best_v, env.P)


# reproduce Abbeel & Klein
env = gym.make('AbbeelKleinGridworld-v0')
V_best_v, pi_best_v = value_iteration(env.P, gamma=0.9)
print(V_best_v)
print_policy(pi_best_v, env.P)
