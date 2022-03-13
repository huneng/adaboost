#ifndef _BINARY_TREE_H_
#define _BINARY_TREE_H_

#include "sample.h"
#include <omp.h>


typedef struct {
    uint8_t x0, y0;
    uint8_t x1, y1;

    uint8_t w, h;
} FeatTemp;


void print_feature_template(FeatTemp *ft, FILE *fout);


typedef struct Node_t{
    float thresh;
    float score;

    FeatTemp ft;

    struct Node_t *lchild;
    struct Node_t *rchild;

    //for debug
    double nw, pw;
    int posSize, negSize;
} Node;


typedef Node Tree;



float predict(Tree *root, int depth, uint32_t *iImg, int stride);
float train(Tree* root, int depth,
        SampleSet *posSet, float *posFeats, double *pws,
        SampleSet *negSet, float *negFeats, double *nws, int featNum);

void print_tree(Tree* root, int depth, FILE *fout);

void save(Tree* root, int depth, FILE *fout);
void load(Tree* root, int depth, FILE *fin);

#endif
