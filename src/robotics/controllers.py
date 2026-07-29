class ProstheticController:
    """Mapea clases de gestos a posiciones de articulaciones (grados/radianes)."""
    def __init__(self):
        # Mapeo: Clase -> {Nombre_Articulación: Ángulo_Objetivo}
        self.gestures = {
            0: {"finger_index": 0.0, "finger_thumb": 0.0, "finger_middle": 0.0},  # Reposo
            1: {"finger_index": 1.2, "finger_thumb": 1.2, "finger_middle": 1.2},  # Puño Cerrado
            2: {"finger_index": 1.0, "finger_thumb": 1.0, "finger_middle": 0.0},  # Pinza
            3: {"finger_index": 0.0, "finger_thumb": 0.0, "finger_middle": 0.0},  # Mano Abierta
            4: {"finger_index": 1.5, "finger_thumb": 0.0, "finger_middle": 0.0},  # Apuntar
        }

    def get_joint_targets(self, gesture_class):
        return self.gestures.get(gesture_class, self.gestures[0])