import numpy as np

# 加载 ADT 数据，shape: (H, W, T)
adt = np.load("adt_1200_00_23_filled.npy")

# 提取第一个时间帧
adt_slice = adt[0,:, :]  # shape: (H, W)

# 创建 land mask
land_mask = np.where(adt_slice == 1e-9, 0, 1).astype(np.uint8)

# 上下翻转（flip along latitude）
#land_mask = np.flipud(land_mask)

# 保存
np.save("land_mask_1200.npy", land_mask)

print("land_mask 生成完成（已翻转），shape:", land_mask.shape)
import matplotlib
matplotlib.use('Agg')  # 不使用图形界面后端

import matplotlib.pyplot as plt

plt.imshow(land_mask, cmap='gray')
plt.title("Land Mask (1 = land, flipped)")
plt.colorbar()
plt.show()
plt.savefig('land_mask.png')
