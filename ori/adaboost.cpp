#include "adaboost.h"


void refine_samples(ObjectDetector *cc, SampleSet *set, int flag);

void init_train_params(TrainParams *params, NegGenerator *generator, int WINW, int WINH, int s){
    generator->dx = 0.1f * WINW;
    generator->dy = 0.1f * WINH;
    generator->tflag = 1;
    generator->maxCount = 20;
    generator->npRate = 1;

    params->recall = 1.0f;
    params->flag = 1;
    params->depth = 4;
    params->treeSize = 256;
    params->times = 10;

    switch(s){
        case 0:
            generator->dx = WINW;
            generator->dy = WINH;
            generator->maxCount = 10;
            generator->tflag = 0;
            generator->npRate = 1;

            params->times = 4;
            params->recall = 0.999;
            break;

        case 1:
            generator->dx = WINW * 0.3;
            generator->dy = WINH * 0.3;
            generator->npRate = 1;
            generator->maxCount = 10;

            params->times = 4;
            break;

        case 2:
            generator->maxCount = 10;


            break;

        case 3:
            params->flag = 0;

            break;

        case 4:
            params->flag = 0;
            params->treeSize = 256;

            break;

        case 5:
            params->flag = 0;
            params->treeSize = 256;

            break;

        default:
            break;
    }
}


int train(ObjectDetector *cc, const char *posFilePath, const char *negFilePath){
    const int WINW = 48;
    const int WINH = 48;

    const int STAGE = 4;

    SampleSet *posSet = NULL, *negSet = NULL;

    TrainParams params;
    NegGenerator generator;

    int ret;

    for(int i = STAGE - 1; i >= 0; i--){
        char filePath[256];

        sprintf(filePath, "cascade_%d.dat", i);

        if(load(cc, filePath) == 0){
            printf("LOAD MODEL %s SUCCESS\n", filePath);
            break;
        }
    }

    if(cc->ssize == 0){
        cc->WINW = WINW;
        cc->WINH = WINH;

        cc->sc = new Forest[STAGE]; assert(cc->sc != NULL);
        memset(cc->sc, 0, sizeof(Forest) * STAGE);
    }
    else {
        Forest *sc = new Forest[STAGE]; assert(cc->sc != NULL);
        memset(sc, 0, sizeof(Forest) * STAGE);
        memcpy(sc, cc->sc, sizeof(Forest) * cc->ssize);

        delete [] cc->sc;
        cc->sc = sc;
    }

    read_samples(posFilePath, 0, WINW, WINH, 0, &posSet);
    print_info(posSet, "pos set");

    negSet = new SampleSet;

    memset(negSet, 0, sizeof(SampleSet));

    negSet->winw = WINW;
    negSet->winh = WINH;

    write_images_into_binary_file(negFilePath, NEG_IMAGES_FILE);

    generator.fin = fopen(NEG_IMAGES_FILE, "rb");
    assert(generator.fin != NULL);

    ret = fread(&generator.isize, sizeof(int), 1, generator.fin);assert(ret == 1);
    generator.id = 0;

    ret = system("mkdir -p model log/neg log/pos");

    for(int s = cc->ssize; s < STAGE; s++){
        printf("---------------- CASCADE %d ----------------\n", s);
        init_train_params(&params, &generator, WINW, WINH, s);

        printf("RECALL = %f, DEPTH = %d, TREE SIZE = %d\n", params.recall, params.depth, params.treeSize);

        generator.scs = cc->sc;
        generator.scSize = s + 1;

        train(cc->sc + s, posSet, negSet, &generator, &params);

        cc->ssize++;

        {
            char filePath[256];
            sprintf(filePath, "model/cascade_%d.dat", s);
            save(cc, filePath);
        }

        {
            char command[256];
            sprintf(command, "mv classifier.txt log/classifier_%d.txt", s);
            ret = system(command);
        }

        printf("---------------------------------------------\n");
    }

    fclose(generator.fin);

    release(&posSet);
    release(&negSet);

    return 0;
}


int predict(ObjectDetector *cc, uint32_t *iImg, int iStride, float &score){
    score = 0;

    for(int i = 0; i < cc->ssize; i++){
        float t;
        if(predict(cc->sc + i, iImg, iStride, t) == 0)
            return 0;

        score = t;
    }

    return 1;
}


void refine_samples(ObjectDetector *cc, SampleSet *set, int flag){
    int ssize = set->ssize;

    printf("refine sample %d, ", ssize);

    if(flag == 0){
        for(int i = 0; i < ssize; i++){
            float score = 0;
            Sample *sample = set->samples[i];

            predict(cc->sc, cc->ssize, sample->iImg, sample->istride, sample->score);
        }
    }
    else {
        for(int i = 0; i < ssize; i++){
            float score = 0;
            Sample *sample = set->samples[i];

            if(predict(cc->sc, cc->ssize, sample->iImg, sample->istride, sample->score) == 0){
                HU_SWAP(set->samples[i], set->samples[ssize - 1], Sample*);

                ssize --;
                i--;
            }
        }
    }

    set->ssize = ssize;

    printf("%d\n", set->ssize);
}



void init_detect_factor(ObjectDetector *cc, float startScale, float endScale, float offset, int layer){
    cc->startScale = startScale;
    cc->endScale = endScale;
    cc->layer = layer;
    cc->offsetFactor = offset;

    float stepFactor = powf(endScale / startScale, 1.0f / (layer - 1));
}


#define MERGE_RECT



int calc_overlapping_area(HRect &rect1, HRect &rect2){
    int cx1 = rect1.x + rect1.width / 2;
    int cy1 = rect1.y + rect1.height / 2;
    int cx2 = rect2.x + rect2.width / 2;
    int cy2 = rect2.y + rect2.height / 2;

    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;

    if(abs(cx1 - cx2) < rect1.width / 2 + rect2.width/2 && abs(cy1 - cy2) < rect1.height / 2 + rect2.height / 2){
        x0 = HU_MAX(rect1.x , rect2.x);
        x1 = HU_MIN(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
        y0 = HU_MAX(rect1.y, rect2.y);
        y1 = HU_MIN(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);
    }
    else {
        return 0;
    }

    return (y1 - y0 + 1) * (x1 - x0 + 1);
}


int merge_rects(HRect *rects, float *confs, int size){
    if(size < 2) return size;

    uint8_t *flags = new uint8_t[size];

    memset(flags, 0, sizeof(uint8_t) * size);

    for(int i = 0; i < size; i++){
        if(flags[i] == 1)
            continue;

        float area0 = 1.0f / (rects[i].width * rects[i].height);

        for(int j = i + 1; j < size; j++){
            if(flags[j] == 1) continue;

            float area1 = 1.0f / (rects[j].width * rects[j].height);

            int overlap = calc_overlapping_area(rects[i], rects[j]);

            if(overlap * area1 > 0.6f || overlap * area0 > 0.6f){
                if(confs[i] > confs[j])
                    flags[j] = 1;
                else
                    flags[i] = 1;
            }
        }
    }

    for(int i = 0; i < size; i++){
        if(flags[i] == 0) {
            continue;
        }

        flags[i] = flags[size - 1];

        rects[i] = rects[size - 1];
        confs[i] = confs[size - 1];

        i --;
        size --;
    }

    delete []flags;
    flags = NULL;

    return size;
}


#include <pthread.h>

#define MAX_BUFFER_SIZE 100
#define MAX_THREAD_NUM 4

typedef struct {
    int tid;

    ObjectDetector *cc;

    uint32_t *iImg;
    int width;
    int height;
    int stride;

    HRect rects[MAX_BUFFER_SIZE];
    float scores[MAX_BUFFER_SIZE];
    int count;

} ThreadParams;


void* thread_call(void *aParams){
    ThreadParams *params = (ThreadParams *)aParams;

    ObjectDetector *cc = params->cc;

    uint32_t *iImg = params->iImg;
    int istride = params->stride;
    int WINW = cc->WINW;
    int WINH = cc->WINH;
    int dx = WINW * cc->offsetFactor;
    int dy = WINH * cc->offsetFactor;

    int lenx = (params->width - WINW) / dx;
    int leny = (params->height - WINH) / dy;

    int maxSize = lenx * leny;
    int count = 0;

    for(int i = params->tid; i < maxSize; i += MAX_THREAD_NUM){
        int y = i / lenx * dy;
        int x = i % lenx * dx;

        float score;

        if(predict(cc->sc, cc->ssize, iImg + y * istride + x, istride, score) == 1){
            HRect rect;

            rect.x = x;
            rect.y = y;
            rect.width = WINW;
            rect.height = WINH;

            params->rects[count] = rect;
            params->scores[count] = score;

            count++;
        }
    }

    params->count = count;
}


int detect_one_scale_threads(ObjectDetector *cc, float scale, uint32_t *iImg, int width, int height, int stride, HRect *resRects, float *resScores)
{
    int WINW = cc->WINW;
    int WINH = cc->WINH;

    int dx = WINW * cc->offsetFactor;
    int dy = WINH * cc->offsetFactor;

    int lenx = (width - WINW) / dx;
    int leny = (height - WINH) / dy;

    int maxSize = lenx * leny;

    assert(maxSize > MAX_THREAD_NUM);

    pthread_t pids[MAX_THREAD_NUM];

    ThreadParams *params = new ThreadParams[MAX_THREAD_NUM];

    for(int i = 0; i < MAX_THREAD_NUM; i++){
        params[i].tid = i;

        params[i].cc = cc;
        params[i].iImg = iImg;
        params[i].width = width;
        params[i].height = height;
        params[i].stride = stride;
        params[i].count = 0;
    }

    for(int i = 0; i < MAX_THREAD_NUM; i++){
        if(pthread_create(pids + i, NULL, thread_call, params + i) != 0){
            printf("Create thread error\n");
            return 0;
        }
    }

    int count = 0;

    for(int i = 0; i < MAX_THREAD_NUM; i++){
        pthread_join(pids[i], NULL);

        int num = params[i].count;

        assert(num < MAX_BUFFER_SIZE);

        memcpy(resRects + count, params[i].rects, sizeof(HRect) * num);
        memcpy(resScores + count, params[i].scores, sizeof(float) * num);

        count += num;
    }

    for(int i = 0; i < count; i++){
        resRects[i].x *= scale;
        resRects[i].y *= scale;
        resRects[i].width *= scale;
        resRects[i].height *= scale;
    }

    delete [] params;

#ifdef MERGE_RECT
    count = merge_rects(resRects, resScores, count);
#endif

    return count;
}


#define FIX_Q 14

int detect_one_scale(ObjectDetector *cc, float scale, uint32_t *iImg, int width, int height, int stride, HRect *resRects, float *resScores){
    int WINW = cc->WINW;
    int WINH = cc->WINH;

    int dx = WINW * cc->offsetFactor;
    int dy = WINH * cc->offsetFactor;

    int count = 0;
    float score;

    int HALF_ONE = 1 << (FIX_Q - 1);
    int FIX_SCALE = scale * (1 << FIX_Q);

    for(int y = 0; y <= height - WINH; y += dy){
        for(int x = 0; x <= width - WINW; x += dx){
            uint32_t *ptr = iImg + y * stride + x;

            if(predict(cc->sc, cc->ssize, iImg + y * stride + x, stride, score) == 1){
                resRects[count].x = (x * FIX_SCALE + HALF_ONE) >> FIX_Q;
                resRects[count].y = (y * FIX_SCALE + HALF_ONE) >> FIX_Q;

                resRects[count].width = (WINW * FIX_SCALE + HALF_ONE) >> FIX_Q;
                resRects[count].height = (WINH * FIX_SCALE + HALF_ONE) >> FIX_Q;

                resScores[count] = score;
                count++;
            }
        }
    }

#ifdef MERGE_RECT
    count = merge_rects(resRects, resScores, count);
#endif

    return count;
}


int calculate_max_size(int width, int height, float startScale, int winSize){
    int minwh = HU_MIN(width, height);

    assert(startScale < 1.0f);

    int size = minwh * startScale;
    float scale = (float)winSize / size;

    width ++;
    height ++;

    if(scale < 1)
        return width * height;

    return (width * scale + 0.5f) * (height * scale + 0.5f);
}

#define BUFFER_SIZE 1000

int detect(ObjectDetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, float **rscores){
    int WINW, WINH, capacity;
    float scale, stepFactor;

    uint8_t *srcImg, *dstImg;
    uint32_t *iImgBuf, *iImg;

    HRect *rects;
    float *scores;
    int top = 0;

    int srcw, srch, srcs, dstw, dsth, dsts;
    int minSide;
    int count;

    int dx = cc->WINW * cc->offsetFactor;
    int dy = cc->WINH * cc->offsetFactor;

    WINW = cc->WINW;
    WINH = cc->WINH;

    scale = cc->startScale;
    stepFactor = powf(cc->endScale / cc->startScale, 1.0f / (cc->layer - 1));

    if(width > height){
        float factor;

        srch = 480;
        srcw = 480 * width / height;
        srcs = srcw;

        factor = WINH / (cc->startScale * height);
        dstw = width * factor + 1;
        dsth = height * factor + 1;
        dsts = dstw;
    }
    else {
        float factor;

        srcw = 480;
        srch = 480 * height / width;
        srcs = srcw;

        factor = WINW / (cc->startScale * width);
        dstw = width * factor + 1;
        dsth = height * factor + 1;
        dsts = dstw;
    }

    srcImg = new uint8_t[srcw * srch]; assert(srcImg != NULL);
    dstImg = new uint8_t[dstw * dsth]; assert(dstImg != NULL);

    iImgBuf = new uint32_t[(dstw + 1) * (dsth + 1)]; assert(iImgBuf != NULL);

    rects  = new HRect[BUFFER_SIZE]; assert(rects != NULL);
    scores = new float[BUFFER_SIZE]; assert(scores != NULL);

    memset(rects, 0, sizeof(HRect) * BUFFER_SIZE);
    memset(scores, 0, sizeof(float) * BUFFER_SIZE);

    resize_gray_image(img, width, height, stride, srcImg, srcw, srch, srcs );

    count = 0;

    minSide = HU_MIN(width, height);

    for(int i = 0; i < cc->layer; i++){
        float scale2 = HU_MIN(WINW, WINH) / (minSide * scale);

        dstw = scale2 * width;
        dsth = scale2 * height;
        dsts = dstw;

        resize_gray_image(srcImg, srcw, srch, srcs, dstImg, dstw, dsth, dsts);

        assert(dstw * dsth < 16777216);

        memset(iImgBuf, 0, sizeof(uint32_t) * (dstw + 1) * (dsth + 1));
        iImg = iImgBuf + dstw + 1 + 1;

        integral_image(dstImg, dstw, dsth, dsts, iImg, dstw + 1);

#if 0
        int lenx = (dstw - WINW) / dx;
        int leny = (dsth - WINH) / dy;

        if(lenx * leny >= MAX_THREAD_NUM)
            count += detect_one_scale_threads(cc, 1.0f / scale2, iImg, dstw, dsth, dstw + 1, rects + count, scores + count);
        else
            count += detect_one_scale(cc, 1.0f / scale2, iImg, dstw, dsth, dstw + 1, rects + count, scores + count);

#else
        count += detect_one_scale(cc, 1.0f / scale2, iImg, dstw, dsth, dstw + 1, rects + count, scores + count);

#endif

        scale *= stepFactor;
    }

    if(count > 0){
#ifdef MERGE_RECT
        count = merge_rects(rects, scores, count);
#endif

        *resRect = new HRect[count]; assert(resRect != NULL);
        memcpy(*resRect, rects, sizeof(HRect) * count);
        *rscores = new float[count]; assert(rscores != NULL);
        memcpy(*rscores, scores, sizeof(float) * count);
    }

    delete [] srcImg;
    delete [] dstImg;
    delete [] iImgBuf;
    delete [] rects;
    delete [] scores;

    return count;
}


int load(ObjectDetector *cc, const char *filePath){
    if(cc == NULL)
        return 1;

    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 2;
    }

    int ret;
    char str[100];

    int versionEpoch = HU_VERSION_EPCH;
    int versionMajor = HU_VERSION_MAJOR;
    int versionMinor = HU_VERSION_MINOR;

    ret = fread(&versionEpoch, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&versionMajor, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&versionMinor, sizeof(int), 1, fin); assert(ret == 1);

    assert(versionEpoch == HU_VERSION_EPCH);
    assert(versionMajor == HU_VERSION_MAJOR);
    assert(versionMinor == HU_VERSION_MINOR);

    ret = fread(&cc->ssize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&cc->WINW, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&cc->WINH, sizeof(int), 1, fin); assert(ret == 1);

    cc->sc = new Forest[cc->ssize]; assert(cc->sc != NULL);

    memset(cc->sc, 0, sizeof(Forest) * cc->ssize);

    for(int i = 0; i < cc->ssize; i++){
        ret = load(cc->sc + i, fin);
        if(ret != 0){
            printf("Load strong classifier error\n");
            fclose(fin);

            delete [] cc->sc;
            delete cc;

            return 2;
        }
    }

    fclose(fin);

    return 0;
}


int save(ObjectDetector *cc, const char *filePath){
    FILE *fout = fopen(filePath, "wb");
    if(fout == NULL)
        return 1;

    int ret;

    char str[100];

    int versionEpoch = HU_VERSION_EPCH;
    int versionMajor = HU_VERSION_MAJOR;
    int versionMinor = HU_VERSION_MINOR;

    ret = fwrite(&versionEpoch, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&versionMajor, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&versionMinor, sizeof(int), 1, fout); assert(ret == 1);

    ret = fwrite(&cc->ssize, sizeof(int), 1, fout);
    ret = fwrite(&cc->WINW, sizeof(int), 1, fout);
    ret = fwrite(&cc->WINH, sizeof(int), 1, fout);

    for(int i = 0; i < cc->ssize; i++){
        ret = save(cc->sc + i, fout);
        if(ret != 0){
            printf("Save strong classifier error\n");
            fclose(fout);
            return 2;
        }
    }

    fclose(fout);

    return 0;
}


void release_data(ObjectDetector *cc){
    if(cc->sc != NULL)
        return;

    for(int i = 0; i < cc->ssize; i++)
        release_data(cc->sc);

    delete [] cc->sc;
    cc->sc = NULL;
}


void release(ObjectDetector **cc){
    if(*cc == NULL)
        return;

    release_data(*cc);
    delete cc;
    cc = NULL;
}
