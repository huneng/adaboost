#ifndef _ADA_BOOST_H_
#define _ADA_BOOST_H_

#include "classifier.h"


typedef struct{
    Forest *sc;
    int WINW, WINH;
    int ssize;

    float startScale;
    float endScale;
    float offsetFactor;

    int layer;
}ObjectDetector;


typedef struct {
    int x, y;
    int width;
    int height;
} HRect;

int train(ObjectDetector *cc, const char *posFilePath, const char *negFilePath);
int predict(ObjectDetector *cc, uint32_t *iImg, int iStride, float &score);

void init_detect_factor(ObjectDetector *cc, float startScale, float endScale, float offset, int layer);
int detect(ObjectDetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, float **rscores);

int save(ObjectDetector *cc, const char *filePath);
int load(ObjectDetector *cascade, const char *filePath);

void release_data(ObjectDetector *cc);
void release(ObjectDetector **cc);

void release_data(NegGenerator *ng);
void release(NegGenerator **ng);

#endif
