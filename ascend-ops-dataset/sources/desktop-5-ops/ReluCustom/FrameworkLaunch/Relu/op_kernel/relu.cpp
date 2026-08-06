#include "kernel_operator.h" //传入核函数头文件
#include <type_traits> 
using namespace AscendC; //命名空间设置为AscendC
constexpr int32_t BUFFER_NUM = 2; //设置双缓冲，提升性能
template<typename TYPE_X> class KernelRelu{ //算子核函数实现
    public:
        __aicore__ inline KernelRelu() {} 
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
            uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, TPipe * pipeIn) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0); //实际输入的总数据量
            this->tileLength = block_size; //每次处理多少数据
            this->ALIGN_NUM = ALIGN_NUM;
            this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); //32B 上取整对齐
            auto startPointer = core_size * GetBlockIdx(); //多核情况下表示外部分数据到不同核的地址，由于本处理器为单核环境，因此该值为0
            auto bufferlength = this->blockLength;
            Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer ,bufferlength);  //设置Globalmemory接收外部输入数据
            Gm_y.SetGlobalBuffer((__gm__ TYPE_X*)y + startPointer , bufferlength);    //设置Globalmemory接收外部输入数据
            this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0); //总的处理次数
            pipe = pipeIn; 
            pipe->InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X)); //声明输入队列x1
            pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_X)); //声明输出队列y
            if constexpr (std::is_same_v<TYPE_X, int16_t>) {  //根据数据类型判断是否需要进行类型转换，是否需要设置缓冲区
                pipeIn->InitBuffer(tmp1, this->tileLength * sizeof(float)); //申请缓冲块tmp1
            }
        }
        __aicore__ inline void Process() { //算子处理函数
            int32_t loopCount = this->tileNum; //总的处理次数
            for (int32_t i = 0; i < loopCount-1; i++) { //前n-1次处理
                CopyIn(i, this->tileLength); //数据搬入
                Compute(i, this->tileLength); //实际计算
                CopyOut(i, this->tileLength); //数据搬出
            }
            uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);    //32B对齐情况下最后一次需要处理的数据量
            CopyIn(loopCount - 1, length);
            Compute(loopCount - 1, length);
            CopyOut(loopCount - 1, length );
        }   
    private:
        __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
           LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>(); //分配输入数据队列x1的变量
           DataCopy(x, Gm_x[progress * this->tileLength], length); //数据从Gmx1->Localtensor,保证数据在AI Core中进行高效并行计算
          
           Q_x.EnQue(x); //x2 数据入队
        }
        __aicore__ inline void Compute(int32_t progress, uint32_t length) { //实际计算逻辑
            LocalTensor<TYPE_X> xLocal = Q_x.DeQue<TYPE_X>(); //x数据出队
            LocalTensor<TYPE_X> yLocal = Q_y.AllocTensor<TYPE_X>(); //分配输出队列y的变量
            if constexpr (std::is_same_v<TYPE_X,int16_t>){ //类型判断是否需要申请临时缓冲区，实际Add已支持四种数据类型，但本操作手册也对类型转换操作进行展示
                LocalTensor<float> p1 = tmp1.Get<float>(); //声明缓冲区变量p1
                Cast(p1,xLocal,AscendC::RoundMode::CAST_NONE, length); //输入数据x1Local进行类型转换 int16->float
    
                Relu(p1,p1,length); //Add加法运算
                Cast(yLocal,p1,AscendC::RoundMode::CAST_FLOOR, length); //输出数据p1->yLocal进行类型转换 float->int16
            }
            else{
                Relu(yLocal,xLocal,length); //half/float直接使用Add API进行运算
            }        
            Q_x.FreeTensor(xLocal);  //释放变量x1Local，走完生命周期
            Q_y.EnQue<TYPE_X>(yLocal);//输出数据yLocal入队
        }
        __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
            LocalTensor<TYPE_X> y = Q_y.DeQue<TYPE_X>();//输出数据yLocal出队
            DataCopy(Gm_y[progress * this->tileLength], y, length); //数据从yLocal->Gm_y,保证计算结果顺利返回外部
            Q_y.FreeTensor(y); //释放输出变量y，走完生命周期
        }
    private:
        TPipe * pipe; //使用TPipe框架进行算子编程
        TQue<QuePosition::VECIN, BUFFER_NUM> Q_x; //输入队列定义
        TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y; //输出队列定义
        TBuf<QuePosition::VECCALC> tmp1; //临时缓冲区定义
        GlobalTensor<TYPE_X> Gm_x,Gm_y; //外部数据定义
        uint32_t blockLength; 
        uint32_t tileNum;
        uint32_t tileLength;
        uint32_t ALIGN_NUM;
    };

extern "C" __global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
 
    GET_TILING_DATA(tiling_data, tiling); //获取host侧的tiling数据
    TPipe pipe; //算子编程框架
    KernelRelu<DTYPE_X> op; //实例化算子对象
    op.Init(x, y,
        tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain,&pipe); //调用初始化函数
    op.Process(); //调用算子处理函数
}