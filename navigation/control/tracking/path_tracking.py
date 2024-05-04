import os
import pickle
import control
import numpy as np
import xml.etree.ElementTree as ET


class PathTracker():

    def __init__(self, config_file):
        # Load XML config file
        tree = ET.parse(config_file)
        config = tree.getroot()

        # Initialize path tracker
        self.read_params(config)

    def read_params(self, config):
        # Read controller parameters
        controller = config.find("controller").find("control")
        self.u_low  = np.array([[float(controller.find("u_low").get("lin"))],
                                [float(controller.find("u_low").get("ang"))]])
        self.u_high = np.array([[float(controller.find("u_high").get("lin"))],
                                [float(controller.find("u_high").get("ang"))]])
        
        # Retrieve dynamics model of vehicle
        controller = config.find("controller").find("dynamics")
        dynamics_file = controller.find("file").get("value")
        self.dynamics = pickle.load(open(dynamics_file, 'rb'))

        # Compile LQR cost matrics
        controller = config.find("controller").find("tracking")
        c_state = np.array([float(controller.find("state_cost").get("x")),
                            float(controller.find("state_cost").get("y")),
                            float(controller.find("state_cost").get("theta")),
                            float(controller.find("state_cost").get("vel")),
                            float(controller.find("state_cost").get("turn"))])
        c_input = np.array([float(controller.find("input_cost").get("lin")),
                            float(controller.find("input_cost").get("ang"))])
        self.Q = np.diag(c_state)
        self.R = np.diag(c_input)

    def compute_state_feedback(self, ref):
        H = len(ref)
        A, B = self.dynamics.get_dynamics_matrices(ref)
        Q, R = self.Q, self.R

        P, K = [Q], []
        for k in range(H-1, -1, -1):
            K.append(np.linalg.inv(R + B[k].T @ P[-1] @ B[k]) @ B[k].T @ P[-1] @ A[k])
            P.append(Q + K[-1].T @ R @ K[-1] + (A[k] - B[k] @ K[-1]).T @ P[-1] @ (A[k] - B[k] @ K[-1]))
        K.reverse()

        return K

