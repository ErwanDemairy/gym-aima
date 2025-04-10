import sys
import io
import numpy as np
import gymnasium as gym
from gymnasium import spaces, utils

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

class AIMAEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "ansi"],  # 렌더링 모드
        "render_fps": 4
    }

    def __init__(
        self,
        render_mode=None,
        map_name="3x4",
        noise=0.2,
        living_rew=0.0,
        sink=False
    ):

        self.render_mode = render_mode
        self.map_name = map_name
        self.noise = noise
        self.living_rew = living_rew
        self.sink = sink

        # 맵 정의
        desc = [
            "FFFG",
            "FWFH",
            "SFFF",
        ]
        self.desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = self.desc.shape
        self.nS = self.nrow * self.ncol  # 총 상태 수
        self.nA = 4                      # 행동 수

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        isd = np.array(self.desc == b'S').astype("float64").ravel()
        isd /= isd.sum()
        self.isd = isd

        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        def to_s(row, col):
            return row * self.ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, self.nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, self.ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                letter = self.desc[row, col]
                for a in range(self.nA):
                    transitions = P[s][a]

                    if self.sink:
                        if letter in b'W':
                            transitions.append((1.0, s, 0, True))
                        elif letter in b'G':
                            transitions.append((1.0, 5, 1, True))
                        elif letter in b'H':
                            transitions.append((1.0, 5, -1, True))
                        else:
                            for b_act in [(a - 1) % 4, a, (a + 1) % 4]:
                                new_row, new_col = inc(row, col, b_act)
                                new_state = to_s(new_row, new_col)
                                new_letter = self.desc[new_row, new_col]
                                if new_letter in b'W':
                                    new_state = s
                                prob = (
                                    1.0 - self.noise
                                    if b_act == a
                                    else self.noise / 2.0
                                )
                                prob = float(np.round(prob, 2))
                                rew = self.living_rew
                                transitions.append((prob, new_state, rew, False))

                    else:
                        if letter in b'GHW':
                            transitions.append((1.0, s, 0, True))
                        else:
                            for b_act in [(a - 1) % 4, a, (a + 1) % 4]:
                                new_row, new_col = inc(row, col, b_act)
                                new_state = to_s(new_row, new_col)
                                new_letter = self.desc[new_row, new_col]
                                if new_letter in b'W':
                                    new_state = s
                                done = (new_letter in b'GH')
                                prob = (
                                    1.0 - self.noise
                                    if b_act == a
                                    else self.noise / 2.0
                                )
                                prob = float(np.round(prob, 2))

                                rew = self.living_rew
                                if new_letter == b'G':
                                    rew += 1.0
                                elif new_letter == b'H':
                                    rew -= 1.0

                                transitions.append((prob, new_state, rew, done))

        self.P = P

        self.s = None
        self.lastaction = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = np.random.choice(self.nS, p=self.isd)
        self.lastaction = None
        return self.s, {}

    def step(self, action):
        transitions = self.P[self.s][action]

        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        p, s_next, reward, done = transitions[idx]

        self.s = s_next
        self.lastaction = action

        return s_next, reward, done, False, {}

    def render(self):
        if self.render_mode == "ansi":
            outfile = io.StringIO()
        else:
            outfile = sys.stdout

        row, col = divmod(self.s, self.ncol)

        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]

        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            actions = ["Left", "Down", "Right", "Up"]
            outfile.write(f"  ({actions[self.lastaction]})\n")
        else:
            outfile.write("\n")

        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if self.render_mode == "ansi":
            return outfile.getvalue()
        else:
            return None