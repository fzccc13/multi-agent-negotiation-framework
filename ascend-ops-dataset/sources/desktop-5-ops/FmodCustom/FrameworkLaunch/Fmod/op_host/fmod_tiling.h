#include "register/tilingdata_base.h"//算子切分注册数据头文件
namespace optiling{
BEGIN_TILING_DATA_DEF(FmodTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);  // 数据分块的最终数量
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM); //根据输入的数据类型计算多少个block才能构成32字节
  TILING_DATA_FIELD_DEF(uint32_t, block_size); // 每次处理的数据量
  TILING_DATA_FIELD_DEF(uint32_t, core_size); // 256B对齐的总输入数量
  TILING_DATA_FIELD_DEF(uint32_t, core_remain); //最后一次处理的数据量
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Fmod,FmodTilingData) //FmodTiling数据注册
}
