#ifndef _TYPE_DEFS_H_
#define _TYPE_DEFS_H_

#define HU_SWAP(x, y, type) {type tmp = (x); (x) = (y); (y) = (tmp);}
#define HU_MIN(i, j) ((i) > (j) ? (j) : (i))
#define HU_MAX(i, j) ((i) < (j) ? (j) : (i))
#define HU_ABS(a) ((a) < 0 ? (-a) : (a))


#define EPSILON 0.000001f
#define HU_PI 3.1415926535

#ifndef HU_FREE
#define HU_FREE(arr) \
{ \
    if(arr != NULL) \
        free(arr); \
    arr = NULL; \
}
#endif



//quick sort implementation, fast than qsort in stdlib.h
#define IMPLEMENT_QSORT(function_name, T, LT)                                   \
    void function_name( T *array, int total_num)                                    \
{                                                                                   \
    int isort_thresh = 7;                                                           \
    int sp = 0;                                                                     \
    \
    struct                                                                          \
    {                                                                               \
        T *lb;                                                                      \
        T *ub;                                                                      \
    }                                                                               \
    stack[48];                                                                      \
    \
    if( total_num <= 1 )                                                            \
    return;                                                                         \
    \
    stack[0].lb = array;                                                            \
    stack[0].ub = array + (total_num - 1);                                          \
    \
    while( sp >= 0 )                                                                \
    {                                                                               \
        T* left = stack[sp].lb;                                                     \
        T* right = stack[sp--].ub;                                                  \
        \
        for(;;)                                                                     \
        {                                                                           \
            int i, n = (int)(right - left) + 1, m;                                  \
            T* ptr;                                                                 \
            T* ptr2;                                                                \
            \
            if( n <= isort_thresh )                                                 \
            {                                                                       \
                insert_sort_##func_name:                                            \
                for( ptr = left + 1; ptr <= right; ptr++ )                          \
                {                                                                   \
                    for( ptr2 = ptr; ptr2 > left && LT(ptr2[0],ptr2[-1]); ptr2--)   \
                    HU_SWAP( ptr2[0], ptr2[-1], T);                            \
                }                                                                   \
                break;                                                              \
            }                                                                       \
            else                                                                    \
            {                                                                       \
                T* left0;                                                           \
                T* left1;                                                           \
                T* right0;                                                          \
                T* right1;                                                          \
                T* pivot;                                                           \
                T* a;                                                               \
                T* b;                                                               \
                T* c;                                                               \
                int swap_cnt = 0;                                                   \
                \
                left0 = left;                                                       \
                right0 = right;                                                     \
                pivot = left + (n/2);                                               \
                \
                if( n > 40 )                                                        \
                {                                                                   \
                    int d = n / 8;                                                  \
                    a = left, b = left + d, c = left + 2*d;                         \
                    left = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))     \
                    : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
                    \
                    a = pivot - d, b = pivot, c = pivot + d;                        \
                    pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))    \
                    : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
                    \
                    a = right - 2*d, b = right - d, c = right;                      \
                    right = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))    \
                    : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
                }                                                                   \
                \
                a = left, b = pivot, c = right;                                     \
                pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))        \
                : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));       \
                if( pivot != left0 )                                                \
                {                                                                   \
                    HU_SWAP( *pivot, *left0, T );                                  \
                    pivot = left0;                                                  \
                }                                                                   \
                left = left1 = left0 + 1;                                           \
                right = right1 = right0;                                            \
                \
                for(;;)                                                             \
                {                                                                   \
                    while( left <= right && !LT(*pivot, *left) )                    \
                    {                                                               \
                        if( !LT(*left, *pivot) )                                    \
                        {                                                           \
                            if( left > left1 )                                      \
                            HU_SWAP( *left1, *left, T );                       \
                            swap_cnt = 1;                                           \
                            left1++;                                                \
                        }                                                           \
                        left++;                                                     \
                    }                                                               \
                    \
                    while( left <= right && !LT(*right, *pivot) )                   \
                    {                                                               \
                        if( !LT(*pivot, *right) )                                   \
                        {                                                           \
                            if( right < right1 )                                    \
                            HU_SWAP( *right1, *right, T);                      \
                            swap_cnt = 1;                                           \
                            right1--;                                               \
                        }                                                           \
                        right--;                                                    \
                    }                                                               \
                    \
                    if( left > right )                                              \
                    break;                                                      \
                    HU_SWAP( *left, *right, T );                                   \
                    swap_cnt = 1;                                                   \
                    left++;                                                         \
                    right--;                                                        \
                }                                                                   \
                \
                if( swap_cnt == 0 )                                                 \
                {                                                                   \
                    left = left0, right = right0;                                   \
                    goto insert_sort_##func_name;                                   \
                }                                                                   \
                \
                n = HU_MIN( (int)(left1 - left0), (int)(left - left1) );           \
                for( i = 0; i < n; i++ )                                            \
                HU_SWAP( left0[i], left[i-n], T );                             \
                \
                n = HU_MIN( (int)(right0 - right1), (int)(right1 - right) );       \
                for( i = 0; i < n; i++ )                                            \
                HU_SWAP( left[i], right0[i-n+1], T );                          \
                n = (int)(left - left1);                                            \
                m = (int)(right1 - right);                                          \
                if( n > 1 )                                                         \
                {                                                                   \
                    if( m > 1 )                                                     \
                    {                                                               \
                        if( n > m )                                                 \
                        {                                                           \
                            stack[++sp].lb = left0;                                 \
                            stack[sp].ub = left0 + n - 1;                           \
                            left = right0 - m + 1, right = right0;                  \
                        }                                                           \
                        else                                                        \
                        {                                                           \
                            stack[++sp].lb = right0 - m + 1;                        \
                            stack[sp].ub = right0;                                  \
                            left = left0, right = left0 + n - 1;                    \
                        }                                                           \
                    }                                                               \
                    else                                                            \
                    left = left0, right = left0 + n - 1;                        \
                }                                                                   \
                else if( m > 1 )                                                    \
                left = right0 - m + 1, right = right0;                          \
                else                                                                \
                break;                                                          \
            }                                                                       \
        }                                                                           \
    }                                                                               \
}




#define RET_OK         0
#define RET_SPEC       1
#define RET_IO_ERROR   2
#define RET_DATA_NULL  3
#define RET_FILE_ERROR 4

#endif
