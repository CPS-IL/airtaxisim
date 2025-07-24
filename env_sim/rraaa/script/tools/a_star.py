"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2

class AStarPlanner:

    def __init__(self, rs):
        """
        rs: robot size[int]
        """
        self.rs = int(rs)
        self.motion = self.get_motion_model()
        self.m = nn.MaxPool2d(kernel_size=self.rs, stride=self.rs)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)


    def planning(self, sx, sy, gx, gy, obs_map):



        torch_obs_map = self.m(torch.FloatTensor(obs_map).unsqueeze(0))
        resize_obs_map = torch_obs_map.squeeze().numpy()

        self.x_width1, self.y_width1 = obs_map.shape
        self.x_width2, self.y_width2 = resize_obs_map.shape

        print(resize_obs_map.shape)

        self.x_width = self.x_width2
        self.y_width = self.y_width2

        sx = int(sx*self.x_width2/self.x_width1)
        sy = int(sy*self.y_width2/self.y_width1)

        gx = int(gx*self.x_width2/self.x_width1)
        gy = int(gy*self.y_width2/self.y_width1)

        start_node = self.Node(sx, sy, 0.0, -1)
        goal_node = self.Node(gx, gy, 0.0, -1)



        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node, resize_obs_map):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        rx = np.int64(np.array(rx)*self.x_width1/self.x_width2).tolist()
        ry = np.int64(np.array(ry)*self.y_width1/self.y_width2).tolist()

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index
        return rx, ry

    def calc_grid_index(self, node):
        return (node.y) * self.x_width + (node.x)

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

    def verify_node(self, node, obs_map):

        if node.x < 0:
            return False
        elif node.y < 0:
            return False
        elif node.x >= self.x_width:
            return False
        elif node.y >= self.y_width:
            return False

        # collision check
        if obs_map[node.x][node.y] == 1:
            return False

        return True


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 104 # [int]
    sy = 107 # [int]
    gx =   0 # [int]
    gy =   0 # [int]
    robot_size = 2 # [int]
    obs_map = np.loadtxt('temp.csv')
    planner = AStarPlanner(robot_size)
    rx, ry = planner.planning(sx, sy, gx, gy, obs_map)

    plt.imshow(obs_map)
    plt.plot(rx, ry, "-r")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")

    plt.show()

if __name__ == '__main__':

    main()
