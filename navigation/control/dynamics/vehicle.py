import os
import numpy as np
import cvxpy as cp
import xml.etree.ElementTree as ET

from navigation.control.dynamics.utils import extract_bags


# Discrete time dynamics of vehicle: x_k+1 = Ax_k + Bu_k
class Vehicle:
    def __init__(self, config_file, bag_files, horizon=1, linear=False):
        # Load XML config file
        tree = ET.parse(config_file)
        config = tree.getroot()
        self.read_params(config)

        # Extract bag file(s)
        if not isinstance(bag_files, list):
            bag_files = [bag_files]
        T, U, X = extract_bags(bag_files, self.cmd_topic, self.odom_topic, self.vehicle_name)

        # Estimate vehicle dyanmics
        self.linear = linear
        self.estimate_dynamics(T, U, X, horizon)

    def read_params(self, config):
        # Read environment parameters
        environment = config.find("environment")
        self.vehicle_name = environment.find("vehicle").get("name")
        self.use_sim = environment.find("use_sim").get("enabled")
        self.use_sim = True if self.use_sim == "True" else False

        # Read ROS topics
        topics = config.find("topics")
        self.odom_topic = topics.find("odom_topic").get("value")
        self.cmd_topic = topics.find("cmd_topic").get("value")

        # Read control parameters
        controller = config.find("controller").find("control")
        self.cmd_rate = float(controller.find("cmd_rate").get("value"))

    def estimate_dynamics(self, T, U, X, H):
        # Estimate matrices of linear discrete dynamics equation
        if self.linear:
            # Get next state of vehicle
            X_NEXT = np.roll(X, -1, axis=1)[:,:-1]
            T, U, X = T[:-1], U[:,:-1], X[:,:-1]

            # Find optimal solution to least squares problem
            ns, ni = np.shape(X)[0], np.shape(U)[0]
            W = np.vstack((X, U))
            AB = cp.Variable((ns, ns + ni))
            objective = cp.Minimize(cp.sum_squares(AB @ W - X_NEXT))
            prob = cp.Problem(objective)
            result = prob.solve()
            if AB.value is not None:
                self.A, self.B = AB.value[:,:ns], AB.value[:,ns:]
                print('Vehicle model matrices: ')
                print('A = ', np.round(self.A, 4))
                print('B = ', np.round(self.B, 4))
            else:
                print('Warning: Unable to estimate linear discrete time model.')

        # Estimate parameters of non-linear discrete dynamics equation
        else:
            # Get future velocities of vehicle
            V, W = X[3,:-H], X[4,:-H]
            U_V = np.array([U[0,i:i+H] for i in range(len(V))])
            U_W = np.array([U[1,i:i+H] for i in range(len(W))])
            V_NEXT = np.roll(X, -H, axis=1)[3,:-H]
            W_NEXT = np.roll(X, -H, axis=1)[4,:-H]

            # Find sub-optimal solution to least squares problem
            v_error, w_error = [], []
            candidates = np.arange(0, 1, 0.01)
            for x in candidates:
                V_PRED = (1 - x)**H * V + x * np.sum([(1 - x)**(H - 1 - k) * U_V[:,k] for k in range(H)], axis=0)
                W_PRED = (1 - x)**H * W + x * np.sum([(1 - x)**(H - 1 - k) * U_W[:,k] for k in range(H)], axis=0)
                v_error.append(np.sum((V_PRED - V_NEXT)**2))
                w_error.append(np.sum((W_PRED - W_NEXT)**2))

            self.a = candidates[np.argmin(v_error)]
            self.b = candidates[np.argmin(w_error)]
            print('Vehicle model parameters: ')
            print('a = ', np.round(self.a, 4))
            print('b = ', np.round(self.b, 4))

    def get_dynamics_matrices(self, x0):
        N = len(x0)

        if self.linear:
            A = np.tile(self.A, (N,1))
            B = np.tile(self.B, (N,1))
        
        else:
            theta = x0[:, 2]
            dt = 1 / self.cmd_rate

            # Compute A matrix
            A = np.tile(np.eye(5), (N,1,1))
            A[:, 0,  3] = dt * np.cos(theta)
            A[:, 1,  3] = dt * np.sin(theta)
            A[:, 2,  4] = dt
            A[:, 3,  3] = 1-self.a
            A[:, 4,  4] = 1-self.b

            # Compute B matrix
            B = np.zeros((N, 5, 2))
            B[:, 3, 0] = self.a
            B[:, 4, 1] = self.b

        return A, B

    def predict_next_states(self, x0, U, update=True):
        N, ns = np.shape(x0)
        N, H, ni = np.shape(U)
        A, B = self.get_dynamics_matrices(x0)

        # Predict next states of vehicle
        x0 = x0[:, :, np.newaxis]
        xpred = np.zeros((N, H, ns))
        for i in range(H):
            u = U[:,i,:][:, :, np.newaxis]
            if update:
                A, B = self.get_dynamics_matrices(x0[:,:,0])
            xpred[:, i, :] = (np.matmul(A, x0) + np.matmul(B, u)).reshape((N,ns))
            x0 = xpred[:, i, :][:, :, np.newaxis]
        return xpred
