MKLROOT=/opt/intel/oneapi/mkl/2023.1.0/
AOCLSPARSE_ROOT=/home/v-yinuoliu/yinuoliu/code/amd-sparse
THREAD=${2:-"48"} 
export OMP_NUM_THREADS=$THREAD
g++ -O3 -DNDEBUG cpu_spmv.cpp -I${AOCLSPARSE_ROOT}/include/ -L${AOCLSPARSE_ROOT}/lib/LP64 -laoclsparse -fopenmp -lrt  -lnuma -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl -o spmv_test
export LD_LIBRARY_PATH=${AOCLSPARSE_ROOT}/lib/LP64:$LD_LIBRARY_PATH
./spmv_test --mtx=/home/v-yinuoliu/yinuoliu/code/SparseCodegen/matrix${1}.mtx --i=100