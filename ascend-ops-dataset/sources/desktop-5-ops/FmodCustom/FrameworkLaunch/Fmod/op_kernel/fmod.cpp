#define K_MAX_SHAPE_DIM 0 
#include "kernel_operator.h" //传入核函数头文件
#include <type_traits>
using namespace AscendC; //命名空间设置为AscendC
constexpr int32_t BUFFER_NUM = 2; //设置双缓冲，提升性能
template<typename TYPE_X> class  KernelFmod {//算子核函数实现
    public:
        __aicore__ inline   KernelFmod () {}
        __aicore__ inline void Init(GM_ADDR x1,GM_ADDR x2, GM_ADDR y,
            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain,
            TPipe * pipeIn) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0); //实际输入的总数据量
            this->tileLength = block_size; //实际输入的总数据量
            this->ALIGN_NUM = ALIGN_NUM;
            this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); //32B 上取整对齐
            auto startPointer = core_size * GetBlockIdx();//多核情况下表示外部分数据到不同核的地址，由于本处理器为单核环境，因此该值为0
            auto bufferlength = this->blockLength;
            Gm_x1.SetGlobalBuffer((__gm__ TYPE_X*)x1 + startPointer ,bufferlength);  //设置Globalmemory接收外部输入数据
            Gm_x2.SetGlobalBuffer((__gm__ TYPE_X*)x2 + startPointer ,bufferlength);  //设置Globalmemory接收外部输入数据
            Gm_y.SetGlobalBuffer((__gm__ TYPE_X*)y + startPointer , bufferlength);    //设置Globalmemory接收外部输入数据
            this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0); //总的处理次数
            pipe = pipeIn;
            pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X)); //声明输入队列x1
            pipe->InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X)); //声明输入队列x2
            pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_X)); //声明输出队列y
            pipe->InitBuffer(tmp1, this->tileLength * sizeof(float));
               if constexpr (std::is_same_v< TYPE_X, half> || std::is_same_v< TYPE_X, bfloat16_t> ){
                pipe->InitBuffer(B_y, this->tileLength * sizeof(float));
                pipe->InitBuffer(B_y2, this->tileLength * sizeof(float));
            }
        }
        __aicore__ inline void Process() {  //算子处理函数
            int32_t loopCount = this->tileNum; //总的处理次数
            for (int32_t i = 0; i < loopCount-1; i++) {//前n-1次处理
                CopyIn(i, this->tileLength); //数据搬入
                Compute(i, this->tileLength); //实际计算
                CopyOut(i, this->tileLength); //数据搬出
            }
            uint32_t length = this->blockLength - this->tileLength * (loopCount - 1); //32B对齐情况下最后一次需要处理的数据量
            CopyIn(loopCount - 1, (length +31 ) / 32 * 32);
            Compute(loopCount - 1, (length +31 ) / 32 * 32);
            CopyOut(loopCount - 1, (length +31 ) / 32 * 32 );
        }   
    private:
        __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
           LocalTensor< TYPE_X> x1 = Q_x1.AllocTensor< TYPE_X >();//分配输入数据队列x1的变量
           LocalTensor< TYPE_X> x2 = Q_x2.AllocTensor< TYPE_X >();//分配输入数据队列x2的变量
           DataCopy(x1, Gm_x1[progress * this->tileLength], length); //数据从Gmx1->Localtensor,保证数据在AI Core中进行高效并行计算
           DataCopy(x2, Gm_x2[progress * this->tileLength], length); //数据从Gmx2->Localtensor,保证数据在AI Core中进行高效并行计算          
           Q_x1.EnQue(x1); //x1 数据入队
           Q_x2.EnQue(x2); //x2 数据入队
        }
        __aicore__ inline void Compute(int32_t progress, uint32_t length) {//实际计算逻辑
            LocalTensor< TYPE_X > x1 = Q_x1.DeQue< TYPE_X >();
            LocalTensor< TYPE_X > x2 = Q_x2.DeQue< TYPE_X >();
            LocalTensor< TYPE_X > y = Q_y.AllocTensor< TYPE_X >();
            if constexpr (std::is_same_v< TYPE_X, half> ) {
                auto p1 = B_y.Get<float>();  //精度转换 half->float
                auto p2 = B_y2.Get<float>();
                auto p3 = tmp1.Get<float>();
                Cast(p1,x1,RoundMode::CAST_NONE, length);
                Cast(p2,x2,RoundMode::CAST_NONE, length);
                Div(p3,p1,p2,length); // p1/p2
                Cast(p3,p3,RoundMode::CAST_TRUNC,length); //trunc(p1/p2)
                Mul(p3,p2,p3,length); // trunc(p1/p2) *p2
                Sub(p1,p1,p3,length); // p1- trunc(p1/p2) *p2
                Cast(y,p1,RoundMode::CAST_NONE,length);  //精度转换 float->half
            }
            else{ //float数据类型分支
                auto p3 = tmp1.Get<float>(); //按照fmod计算公式直接计算
                Div(p3,x1,x2,length);
                Cast(p3,p3,RoundMode::CAST_TRUNC,length);
                Mul(p3,p3,x2,length);
                Sub(y,x1,p3,length);
            }
            Q_x1.FreeTensor(x1);  //释放变量x1Local，走完生命周期
            Q_x2.FreeTensor(x2);  //释放变量x2Local，走完生命周期
            Q_y.EnQue<TYPE_X>(y);//输出数据yLocal入队
        }
        __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
            LocalTensor<TYPE_X> y = Q_y.DeQue<TYPE_X>();//输出数据yLocal出队
            DataCopy(Gm_y[progress * this->tileLength], y, length); //数据从yLocal->Gm_y,保证计算结果顺利返回外部
            Q_y.FreeTensor(y); //释放输出变量y，走完生命周期
        }
    private:
        TPipe * pipe; //使用TPipe框架进行算子编程
        TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1,Q_x2; //输入队列定义
        TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y; //输出队列定义                   
        TBuf<QuePosition::VECCALC> B_y,B_y2,tmp1; //临时缓冲区定义
        GlobalTensor<TYPE_X> Gm_x1,Gm_x2, Gm_y; //外部数据定义
        uint32_t blockLength;
        uint32_t tileNum;
        uint32_t tileLength;
        uint32_t ALIGN_NUM;
    };
extern "C" __global__ __aicore__ void fmod(GM_ADDR x1,GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) { //外部fmod函数，整个程序的入口
    GET_TILING_DATA(tiling_data, tiling); //获取host侧的tiling数据
    TPipe pipe; //算子编程框架
    KernelFmod<DTYPE_X1> op; //实例化算子对象
    op.Init(x1,x2, y, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,&pipe); //调用初始化函数
    op.Process();//调用算子处理函数
}
