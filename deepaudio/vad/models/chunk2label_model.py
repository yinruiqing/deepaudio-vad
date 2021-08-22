from .base import BaseModel, Resolution


class Chunk2LabelModel(BaseModel):
    def on_save_checkpoint(self, checkpoint):
        checkpoint["configs"] = self.configs
        checkpoint["Resoultion"] = Resolution.CHUNK
