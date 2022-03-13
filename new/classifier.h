#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "tree.h"
#include <omp.h>


typedef struct {
    Tree **trees;

    int treeSize;

    int capacity;
    int depth;

    float thresh;
} StrongClassifier;


typedef struct{
    float recall;
    float precision;

    int featNum;
    int depth;
} TrainParams;

int train(StrongClassifier *sc, SampleSet *posSet, SampleSet *negSet, TrainParams *params);
int predict(StrongClassifier *sc, uint32_t *intImg, int istride, float &score);
int predict(StrongClassifier *sc, int scSize, uint32_t *intImg, int istride, float &score);

int save(StrongClassifier *sc, FILE *fout);
int load(StrongClassifier *sc, FILE *fin);

void get_feature_template(StrongClassifier *scs, int scSize, FeatTemp **fts, int &top);

void release(StrongClassifier **sc);
void release_data(StrongClassifier *sc);

#endif
