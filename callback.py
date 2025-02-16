from transformers import TrainerCallback

class PushToHubCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 1000 == 0:

            kwargs["model"].push_to_hub(
                repo_id="gpt-bi",
                organization="AuriLab",
                commit_message=f"Checkpoint a los {state.global_step} steps"
            )