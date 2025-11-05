import os
import xarray as xr
import numpy as np
from scipy.io import savemat
from tqdm import tqdm

# 文件夹路径
data_dir = r"E:\Sea_Surface_Height\adt"
file_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")])

# 存储结果的列表
ugos_list = []
time_list = []

# 设定经纬度范围
lon_min, lon_max = 118, 158
lat_min, lat_max = 1, 31

# 读取第一个文件以获取经纬度索引
with xr.open_dataset(file_list[0]) as ds:
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_sel = lon[lon_inds]
    lat_sel = lat[lat_inds]

# 遍历所有文件提取数据
for file in tqdm(file_list, desc="Processing"):
    try:
        with xr.open_dataset(file) as ds:
            ugos = ds['ugos'].isel(latitude=lat_inds, longitude=lon_inds).squeeze().values
            ugos = ugos.astype(np.float32)
            ugos = np.where(ugos == -214748.3647, np.nan, ugos)  # 处理FillValue
            ugos_list.append(ugos)
            time_list.append(ds['time'].values[0])
    except Exception as e:
        print(f"Error processing {file}: {e}")

# 转换为 numpy 数组
ugos_array = np.stack(ugos_list)  # shape: (time, lat, lon)
time_array = np.array(time_list)

# 保存为 .mat 文件
savemat("ugos_small_1200_00_23.mat", {
    "ugos": ugos_array
})
