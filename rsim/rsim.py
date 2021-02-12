import robosim
import numpy as np
from typing import Dict, List
from Entities import *
from rsim.Render_gym import RCGymRender


def clip(val, vmin, vmax):
    return min(max(val, vmin), vmax)

def normX(x):
    return clip(x / 0.85, -1.2, 1.2)


def normVx(v_x):
    return clip(v_x / 0.8, -1.25, 1.25)


def normVt(vt):
    return clip(vt / 573, -1.2, 1.2)

class RSim:
    def __init__(self, field_type: int = 0, n_robots_blue: int =3,
                 n_robots_yellow: int=3, time_step_ms: int=16):
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.view = None

        # Positions needed just to initialize the simulator
        ball_pos = [0, 0, 0, 0]
        blue_robots_pos = [[-0.2 * i, 0, 0]
                           for i in range(1, n_robots_blue + 1)]
        yellow_robots_pos = [[0.2 * i, 0, 0]
                             for i in range(1, n_robots_yellow + 1)]

        self.simulator = self._init_simulator(field_type=field_type,
                                              n_robots_blue=n_robots_blue,
                                              n_robots_yellow=n_robots_yellow,
                                              ball_pos=ball_pos,
                                              blue_robots_pos=blue_robots_pos,
                                              yellow_robots_pos=yellow_robots_pos,
                                              time_step_ms=time_step_ms)
        
        self.field_params = self.get_field_params()

    def reset(self, frame: Frame):
        placement_pos = self._placement_dict_from_frame(frame)
        self.simulator.reset(**placement_pos)

    def stop(self):
        del(self.simulator)

    def send(self, commands):
        sim_commands = np.zeros(
            (self.n_robots_blue + self.n_robots_yellow, 2), dtype=np.float64)

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id
            # convert from linear speed to angular speed
            sim_commands[rbt_id][0] = cmd.v_wheel1
            sim_commands[rbt_id][1] = cmd.v_wheel2
        self.simulator.step(sim_commands)
        self.render()

    def receive(self) -> Frame:
        state = self.simulator.get_state()
        # Update frame with new state
        self.frame = Frame()
        self.frame.parse(state)

        return self._frame_to_observations()

    def get_field_params(self):
        return self.simulator.get_field_params()
    
    def _placement_dict_from_frame(self, frame: Frame):
        replacement_pos: Dict[str, np.ndarray] = {}

        ball_pos: List[float] = [frame.ball.x, frame.ball.y,
                                 frame.ball.v_x, frame.ball.v_y]
        replacement_pos['ball_pos'] = np.array(ball_pos)

        blue_pos: List[List[float]] = []
        for robot in frame.robots_blue.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            blue_pos.append(robot_pos)
        replacement_pos['blue_robots_pos'] = np.array(blue_pos)

        yellow_pos: List[List[float]] = []
        for robot in frame.robots_yellow.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            yellow_pos.append(robot_pos)
        replacement_pos['yellow_robots_pos'] = np.array(yellow_pos)

        return replacement_pos

    def _init_simulator(self, field_type, n_robots_blue, n_robots_yellow,
                        ball_pos, blue_robots_pos, yellow_robots_pos,
                        time_step_ms):

        return robosim.SimulatorVSS(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            ball_pos=ball_pos,
            blue_robots_pos=blue_robots_pos,
            yellow_robots_pos=yellow_robots_pos,
            time_step_ms=time_step_ms
            )

    def _frame_to_observations(self):

        observation = []

        observation.append(normX(self.frame.ball.x))
        observation.append(normX(self.frame.ball.y))
        observation.append(normVx(self.frame.ball.v_x))
        observation.append(normVx(self.frame.ball.v_y))

        for i in range(3):
            observation.append(normX(self.frame.robots_blue[i].x))
            observation.append(normX(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(normVx(self.frame.robots_blue[i].v_x))
            observation.append(normVx(self.frame.robots_blue[i].v_y))
            observation.append(normVt(self.frame.robots_blue[i].v_theta))

        for i in range(3):
            observation.append(normX(self.frame.robots_yellow[i].x))
            observation.append(normX(self.frame.robots_yellow[i].y))
            observation.append(normVx(self.frame.robots_yellow[i].v_x))
            observation.append(normVx(self.frame.robots_yellow[i].v_y))
            observation.append(normVt(self.frame.robots_yellow[i].v_theta))
            
        return np.array(observation, dtype=np.float32)
    
    def render(self, mode = None) -> None:
        '''
        Renders the game depending on 
        ball's and players' positions.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        if self.view == None:
            self.view = RCGymRender(self.n_robots_blue,
                                self.n_robots_yellow,
                                self.field_params,
                                simulator='vss')

        self.view.render_frame(self.frame)
        