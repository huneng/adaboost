#ifndef _BINARY_TREE_H_
#define _BINARY_TREE_H_

#include "sample.h"

typedef struct {
    uint8_t x00, y00;
    uint8_t x01, y01;
    uint8_t x10, y10;
    uint8_t x11, y11;
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

float train(Tree* root, int depth, SampleSet *posSet, SampleSet *negSet, float recall, int featTimes);

void print_tree(Tree* root, int depth, FILE *fout);

void save(Tree* root, int depth, FILE *fout);
void load(Tree* root, int depth, FILE *fin);

#endif
