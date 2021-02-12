import socket
import binascii
import numpy as np
import pb_fira.packet_pb2 as packet_pb2
from pb_fira.state_pb2 import *
import pb_fira.command_fira_pb2 as command_pb2
import pb_fira.common_pb2 as common_pb2
import pb_fira.packet_pb2 as packet_pb2
import pb_fira.replacement_pb2 as replacement_pb2

from Entities import *

def clip(val, vmin, vmax):
    return min(max(val, vmin), vmax)

def normX(x):
    return clip(x / 0.85, -1.2, 1.2)


def normVx(v_x):
    return clip(v_x / 0.8, -1.25, 1.25)


def normVt(vt):
    return clip(vt / 573, -1.2, 1.2)

class FiraClient:

    def __init__(self, 
            vision_ip='224.0.0.1',
            vision_port=10002, 
            cmd_ip='127.0.0.1',
            cmd_port=20011):
        """
        Init SSLClient object.
        Extended description of function.
        Parameters
        ----------
        ip : str
            Multicast IP in format '255.255.255.255'. 
        port : int
            Port up to 1024. 
        """

        self.vision_ip = vision_ip
        self.vision_port = vision_port        
        self.com_ip = cmd_ip
        self.com_port = cmd_port
        self.com_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.com_address = (self.com_ip, self.com_port)

        self.vision_sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.vision_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.vision_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.vision_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.vision_sock.bind((self.vision_ip, self.vision_port))

    def receive(self):
        """Receive package and decode."""

        data, _ = self.vision_sock.recvfrom(1024)
        decoded_data = packet_pb2.Environment().FromString(data)
        self.frame = FramePB()
        self.frame.parse(decoded_data)
        
        return self._frame_to_observations()

    def send(self, commands):        
        # prepare commands
        pkt = packet_pb2.Packet()
        d = pkt.cmd.robot_commands

        # send wheel speed commands for each robot
        for cmd in commands:
            robot = d.add()
            robot.id = cmd.id
            robot.yellowteam = cmd.yellow

            # convert from linear speed to angular speed
            robot.wheel_left = cmd.v_wheel1
            robot.wheel_right = cmd.v_wheel2

        # send commands
        data = pkt.SerializeToString()
        self.com_socket.sendto(data, self.com_address)
        
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
            
        return np.array(observation)
        