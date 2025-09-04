import numpy as np
import matplotlib.pyplot as plt

class CourtHeatmap:
    def __init__(self, width=800, height=400, bins_x=16, bins_y=8):
        self.width = width
        self.height = height
        self.bins_x = bins_x
        self.bins_y = bins_y
        self.grid = np.zeros((bins_y, bins_x), dtype=np.float32)

    def add(self, x, y):
        bx = int(np.clip(x / self.width * self.bins_x, 0, self.bins_x - 1))
        by = int(np.clip(y / self.height * self.bins_y, 0, self.bins_y - 1))
        self.grid[by, bx] += 1.0

    def render(self, out_path="heatmap.png"):
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(self.grid, origin='lower',
                   extent=[0, self.width, 0, self.height],
                   aspect='auto', alpha=0.9)
        plt.colorbar(label='Shot contacts')
        plt.title("Badminton Shot Contact Heatmap")
        plt.xlabel("Court X")
        plt.ylabel("Court Y")
        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path
