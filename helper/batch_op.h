template <typename NE_list_Iter,typename Index_list_Iter,typename NV_list_Iter,typename Bias_T,typename ValueT>
inline void compute_before_scatter_auto_gen( Bias_T idx_global,
NV_list_Iter vector_x,
NE_list_Iter values,
Index_list_Iter column_indices,
ValueT* scatter_input_B_einsum) {
for (int batch_id_o = 0; batch_id_o < BATCH_SIZE; batch_id_o++)
{
    for (int batch_id = 0; batch_id < BATCH_SIZE/*another dim*/; batch_id++)
    {
        scatter_input_B_einsum[batch_id_o] += values[batch_id * 89306020 + idx_global] * vector_x[batch_id * 1102824 + column_indices[idx_global]];
    }
}
}