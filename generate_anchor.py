import numpy as np
from config import config
anchor_num = len(config.anchor_ratios) * len(config.anchor_scales)
anchor = np.zeros((anchor_num, 4), dtype=np.float32)
size = 8 * 8
count = 0
for ratio in config.anchor_ratios:
    # ws = int(np.sqrt(size * 1.0 / ratio))
    ws = int(np.sqrt(size / ratio))
    hs = int(ws * ratio)
    for scale in config.anchor_scales:
        wws = ws * scale
        hhs = hs * scale
        anchor[count, 0] = 0
        anchor[count, 1] = 0
        anchor[count, 2] = wws
        anchor[count, 3] = hhs
        count += 1

anchor = np.tile(anchor, 17 * 17).reshape((-1, 4))
# (5,4x225) to (225x5,4)
ori = - (17 // 2) * 8
# the left displacement
xx, yy = np.meshgrid([ori + 8 * dx for dx in range(17)],
                     [ori + 8 * dy for dy in range(17)])
# (15,15)
xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
         np.tile(yy.flatten(), (anchor_num, 1)).flatten()
# (15,15) to (225,1) to (5,225) to (225x5,1)
anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
print(anchor)