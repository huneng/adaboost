#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "tree.h"
#include <omp.h>

#define NEG_IMAGES_FILE "neg_images.bin"

typedef struct {
    float recall;

    int treeSize;
    int depth;
    int flag;
    int times;
} TrainParams;


typedef struct {
    Tree **trees;

    int treeSize;

    int capacity;
    int depth;

    float *threshes;
} Forest;


typedef struct{
    FILE *fin;
    int isize;
    int id;

    Forest *scs;
    int scSize;

    int dx, dy;
    int tflag;

    int npRate;

    int maxCount;
} NegGenerator;


void train(Forest *sc, SampleSet *posSet, SampleSet *negSet, NegGenerator *generator, TrainParams *params);

int predict(Forest *sc, uint32_t *intImg, int istride, float &score);
int predict(Forest *sc, int scSize, uint32_t *intImg, int istride, float &score);

int save(Forest *sc, FILE *fout);
int load(Forest *sc, FILE *fin);

void release(Forest **sc);
void release_data(Forest *sc);

#endif
