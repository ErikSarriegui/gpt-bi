"""
==============================
CALLBACK
==============================

Erik Sarriegui Perez, AuriLab, Feb 2025
"""
from transformers import TrainerCallback

class PushToHubCallback(TrainerCallback):
    """
    Callback para enviar checkpoints del modelo a Hugging Face Hub durante el entrenamiento.

    Este callback se encarga de subir el modelo a Hugging Face Hub cada cierto número de pasos (`push_steps`).  Permite mantener un registro del progreso del entrenamiento y facilita la recuperación de modelos en puntos específicos.

    Args:
        repo_id (str): El ID del repositorio en Hugging Face Hub.
        organization (str): La organización a la que pertenece el repositorio (puede ser None).
        push_steps (int): La frecuencia (en pasos de entrenamiento) con la que se enviarán los checkpoints.
    """
    def __init__(self, repo_id : str, organization : str, push_steps : int) -> None:
        self.repo_id = repo_id
        self.organization = organization
        self.push_steps = push_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.push_steps == 0:

            kwargs["model"].push_to_hub(
                repo_id = self.repo_id,
                organization = self.organization,
                commit_message=f"Checkpoint a los {state.global_step} steps"
            )