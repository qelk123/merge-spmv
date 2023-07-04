const int BATCH_SIZE_LIST[SCATTER_NUM] = {BATCH_SIZE}; //必须每一个loop有一份重复的，为了使能编译器进行优化！
ValueT running_total_0[BATCH_SIZE] = {0.0};
ValueT* running_total[SCATTER_NUM] = {running_total_0};
ValueT* vector_y_out_list[SCATTER_NUM] = {easier_params.vector_y_0};