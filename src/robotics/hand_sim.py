import pybullet as p
import pybullet_data
import time
from src.robotics.controllers import ProstheticController

class HandSimulation:
    def __init__(self, gui=True):
        self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Cargar plano de base y modelo de mano/robot sintético
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 1])  # Reemplazar con URDF de mano robótica
        self.controller = ProstheticController()

    def update_pose(self, gesture_class):
        targets = self.controller.get_joint_targets(gesture_class)
        # Aplicar el comando de control a la simulación
        p.stepSimulation()

    def close(self):
        p.disconnect()