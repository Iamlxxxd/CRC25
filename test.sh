#!/bin/bash

# 获取脚本所在目录的上一级作为基础目录
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $BASE_DIR
ROUTE_NAME="osdpm_1_1"
echo $ROUTE_NAME
# 硬编码设置所有必要的路径
META_DATA_PATH="$BASE_DIR/data/train/routes/$ROUTE_NAME/metadata.json"
BASIC_NETWORK_PATH="$BASE_DIR/data/train/maps/osdpm_segment_1.gpkg"  #这里需要注意下读的不一定是这个图
FOIL_JSON_PATH="$BASE_DIR/data/train/routes/$ROUTE_NAME/foil_route.json"
DF_PATH_FOIL_PATH="$BASE_DIR/data/train/routes/$ROUTE_NAME/foil_route.gpkg"
GDF_COORDS_PATH="$BASE_DIR/data/train/routes/$ROUTE_NAME/route_start_end.csv"
OUTPUT_PATH="$BASE_DIR/my_demo/output/submission_out"

# 创建输出目录
mkdir -p $OUTPUT_PATH

# 调用submission_template.py
python submission_template.py \
  --meta_data_path $META_DATA_PATH \
  --basic_network_path $BASIC_NETWORK_PATH \
  --foil_json_path $FOIL_JSON_PATH \
  --df_path_foil_path $DF_PATH_FOIL_PATH \
  --gdf_coords_path $GDF_COORDS_PATH \
  --output_path $OUTPUT_PATH

echo "测试完成！结果保存在: $OUTPUT_PATH"
