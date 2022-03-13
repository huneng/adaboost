#include "adaboost.h"


#define NEG_IMAGES_FILE "neg_images.bin"


int generate_negative_images(const char *listFile, const char *outfile){
    FILE *fin = fopen(outfile, "rb");
    if(fin != NULL){
        fclose(fin);
        return 0;
    }

    std::vector<std::string> imgList;
    read_file_list(listFile, imgList);

    int size = imgList.size();

    printf("GENERATE NEGATIVE IMAGES %ld\n", imgList.size());
    FILE *fout = fopen(outfile, "wb");
    if(fout == NULL){
        printf("Can't open file %s\n", outfile);
        return 1;
    }

    char rootDir[256], fileName[256], ext[30];
    int ret;

    ret = fwrite(&size, sizeof(int), 1, fout), assert(ret == 1);

    for(int i = 0; i < size; i++){
        const char *imgPath = imgList[i].c_str();
        cv::Mat img = cv::imread(imgPath, 0);

        if(img.empty()){
            printf("Can't open image %s\n", imgPath);
            continue;
        }

        analysis_file_path(imgPath, rootDir, fileName, ext);

        ret = fwrite(fileName, sizeof(char), 255, fout); assert(ret == 255);

        if(img.cols > img.rows && img.rows > 720)
            cv::resize(img, img, cv::Size(img.cols * 720 / img.rows, 720));
        else if(img.rows > img.cols && img.cols > 720)
            cv::resize(img, img, cv::Size(720, 720 * img.rows / img.cols));

        if(img.rows * img.cols > 1024 * 1024){
            float scale = 1024.0f / img.rows;
            scale = HU_MIN(1024.0f / img.cols, scale);
            cv::resize(img, img, cv::Size(scale * img.cols, scale * img.rows));
        }

        assert(img.cols * img.rows <= 1024 * 1024);

        ret = fwrite(&img.rows, sizeof(int), 1, fout); assert(ret == 1);
        ret = fwrite(&img.cols, sizeof(int), 1, fout); assert(ret == 1);

        if(img.cols == img.step)
            ret = fwrite(img.data, sizeof(uint8_t), img.rows * img.cols, fout);
        else
            for(int y = 0; y < img.rows; y++)
                ret = fwrite(img.data + y * img.step, sizeof(uint8_t), img.cols, fout);
        if(i % 10000 == 0)
            printf("%d ", i), fflush(stdout);
    }

    printf("\n");
    fclose(fout);

    printf("IMAGE SIZE: %d\n", size);
    return 0;
}


void load_images(FILE *fin, uint8_t **imgs, int *widths, int *heights, char **fileNames, int size, int tflag){
    int ret = 0;

    uint8_t *buffer = new uint8_t[4096 * 4096];

    for(int j = 0; j < size; j++){
        int width, height;

        ret = fread(fileNames[j], sizeof(char), 255, fin); assert(ret == 255);
        ret = fread(heights + j, sizeof(int), 1, fin); assert(ret == 1);
        ret = fread(widths + j, sizeof(int), 1, fin); assert(ret == 1);

        ret = fread(imgs[j], sizeof(uint8_t), widths[j] * heights[j], fin); assert(ret == widths[j] * heights[j]);

        if(tflag)
            transform_image(imgs[j], widths[j], heights[j], widths[j], buffer);

        /*
        cv::Mat sImg(heights[j], widths[j], CV_8UC1, imgs[j]);
        cv::imshow("sImg", sImg);
        cv::waitKey();
        //*/
    }

    delete [] buffer;
}


int detect_image(uint8_t *img, int width, int height, int stride, char *fileName,
        NegGenerator *generator, uint8_t *buffer, uint32_t *intImg, SampleSet *set, int maxSize){
    int count = 0;
    float scale = 1.0f;

    int WINW = set->winw;
    int WINH = set->winh;
    int sq = (WINW + 1) * (WINH + 1);

    int dx = generator->dx;
    int dy = generator->dy;

    uint32_t *iImg;
    int istride;

    for(int l = 0; l < 20; l++){
        int w, h, s;

        if(width > height){
            h = WINH * scale;
            w = h * width / height;
            w = HU_MAX(w, WINW);

        }
        else {
            w = WINW * scale;
            h = w * height / width;
            h = HU_MAX(h, WINH);
        }

        if(w * h > 4096 * 4096) break;

        s = w;
        resizer_bilinear_gray(img, width, height, stride, buffer, w, h, w);

        memset(intImg, 0, sizeof(uint32_t) * (w + 1) * (h + 1));

        iImg = intImg + w + 1 + 1;
        istride = w + 1;

        integral_image(buffer, w, h, w, iImg, istride);

        for(int y = 0; y <= h - WINH; y += dy){
            for(int x = 0; x <= w - WINW; x += dx){
                uint32_t *ptr = iImg + y * istride + x;
                float score;

                if(predict(generator->scs, generator->scSize, ptr, istride, score) == 1){
                    char filePath[256];
                    sprintf(filePath, "%s_%d", fileName, count++);

                    Sample *sample = new Sample;
                    memset(sample, 0, sizeof(Sample));

                    create_sample(sample, buffer + y * w + x, WINW, WINH, w, filePath);
                    sample->score = score;

                    add_sample_capacity_unchange(set, sample);
                }
            }
        }

        if(count > maxSize)
            break;

        scale *= 1.12f;
    }

    return count;
}


void generate_negative_samples(SampleSet *negSet, NegGenerator *generator, int needSize){
    if(needSize <= 0)
        return;

    int WINW = negSet->winw;
    int WINH = negSet->winh;

    const int BUF_SIZE = 1000;

    uint8_t **imgs;
    char **fileNames;
    int *widths, *heights;

    imgs = new uint8_t*[BUF_SIZE];
    widths  = new int[BUF_SIZE];
    heights = new int[BUF_SIZE];
    fileNames = new char*[BUF_SIZE];

    for(int i = 0; i < BUF_SIZE; i++){
        fileNames[i] = new char[256];
        imgs[i] = new uint8_t[1024 * 1024];
    }

    int threadNum = omp_get_num_procs() - 1;

    SampleSet *sets = new SampleSet[threadNum];
    uint8_t **sImgs = new uint8_t*[threadNum];
    uint32_t **intImgs = new uint32_t*[threadNum];

    memset(sets, 0, sizeof(SampleSet) * threadNum);

    for(int i = 0; i < threadNum; i++){
        sImgs[i] = new uint8_t[4096 * 4096];
        intImgs[i] = new uint32_t[4097 * 4097];

        sets[i].winw = negSet->winw;
        sets[i].winh = negSet->winh;
        sets[i].ssize = 0;

        reserve(sets + i, (needSize / threadNum + 1) * 2.0f);
    }

    int id = generator->id;
    int ret;
    int ssize;

    //printf("%d %d %d\n", generator->isize, generator->id, generator->fsize);
    while(1){
        printf("%d ", id);
        int ei = HU_MIN(BUF_SIZE, generator->isize - id);

        //printf("Load images %d, ", ei);fflush(stdout);
        load_images(generator->fin, imgs, widths, heights, fileNames, ei, generator->tflag);

        //printf("sample, "); fflush(stdout);
#pragma omp parallel for num_threads(threadNum)
        for(int i = 0; i < ei; i++){
            int tId = omp_get_thread_num();
            assert(tId < threadNum);

            uint8_t *img = imgs[i];

            int width = widths[i];
            int height = heights[i];

            char *fileName = fileNames[i];

            uint8_t *sImg = sImgs[tId];
            uint32_t *intImg = intImgs[tId];
            SampleSet *set = sets + tId;

            int count = detect_image(img, width, height, width, fileName, generator, sImg, intImg, set, generator->maxCount);
        }

        ssize = 0;
        for(int i = 0; i < threadNum; i++)
            ssize += sets[i].ssize;

        id += BUF_SIZE;

        if(id >= generator->isize){
            id = 0;
            fclose(generator->fin);
            generator->fin = fopen(NEG_IMAGES_FILE, "rb"); assert(generator->fin != NULL);
            ret = fread(&generator->isize, sizeof(int), 1, generator->fin);
            assert(ret == 1);
        }

        printf("%d, ", ssize);fflush(stdout);

        if(ssize > needSize)
            break;
    }
    printf("\n");

    generator->id = id;

    ssize += negSet->ssize;
    reserve(negSet, ssize);

    for(int i = 0; i < threadNum; i++){
        memcpy(negSet->samples + negSet->ssize, sets[i].samples, sizeof(Sample*) * sets[i].ssize);
        negSet->ssize += sets[i].ssize;

        delete [] sets[i].samples;

        sets[i].samples = NULL;
        sets[i].ssize = 0;
        sets[i].capacity = 0;

        delete [] sImgs[i];
        delete [] intImgs[i];
    }

    delete [] sets;
    delete [] sImgs;
    delete [] intImgs;

    for(int i = 0; i < BUF_SIZE; i++){
        delete [] imgs[i];
        delete [] fileNames[i];
    }

    delete [] imgs;
    delete [] fileNames;
    delete [] widths;
    delete [] heights;
}



void refine_samples(Cascade *cc, SampleSet *set, int flag);


void init_train_params(TrainParams *params, NegGenerator *generator, int WINW, int WINH, int s){
    params->recall = 1.0f;
    params->precision = 0.95f;
    params->featNum = 500;
    params->depth = 4;

    generator->maxCount = 20;
    generator->dx = 0.1 * WINW;
    generator->dy = 0.1 * WINH;
    generator->npRate = 1;
    generator->tflag = 1;

    switch(s){
        case 0:
            params->recall = 0.9999f;
            params->precision = 0.6f;

            generator->maxCount = 10;
            generator->dx = WINW;
            generator->dy = WINH;

            break;
        case 1:
            params->precision = 0.8f;

            generator->maxCount = 10;
            generator->dx = WINW * 0.5f;
            generator->dy = WINH * 0.5f;

            break;
        case 2:
            params->precision = 0.9f;

            generator->maxCount = 10;

            break;

        case 3:
            break;
    }
}


int train(Cascade *cc, const char *posFilePath, const char *negFilePath){
    const int WINW = 64;
    const int WINH = 64;

    const int STAGE = 6;

    SampleSet *posSet = NULL, *negSet = NULL;

    TrainParams params;
    NegGenerator generator;

    int ret;

    //load trained model
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

        cc->sc = new StrongClassifier[STAGE]; assert(cc->sc != NULL);
        memset(cc->sc, 0, sizeof(StrongClassifier) * STAGE);
    }
    else {
        StrongClassifier *sc = new StrongClassifier[STAGE]; assert(cc->sc != NULL);
        memset(sc, 0, sizeof(StrongClassifier) * STAGE);
        memcpy(sc, cc->sc, sizeof(StrongClassifier) * cc->ssize);

        delete [] cc->sc;
        cc->sc = sc;
    }

    //load positive samples
    read_samples(posFilePath, 0, WINW, WINH, 0, &posSet);
    print_info(posSet, "pos set");

    negSet = new SampleSet;

    memset(negSet, 0, sizeof(SampleSet));

    negSet->winw = WINW;
    negSet->winh = WINH;

    //initialize negative sample dataset
    generate_negative_images(negFilePath, NEG_IMAGES_FILE);

    generator.fin = fopen(NEG_IMAGES_FILE, "rb");
    assert(generator.fin != NULL);

    ret = fread(&generator.isize, sizeof(int), 1, generator.fin);assert(ret == 1);
    generator.id = 0;

    ret = system("mkdir -p model log/neg log/pos");

    for(int s = cc->ssize; s < STAGE; s++){
        printf("---------------- CASCADE %d ----------------\n", s);
        init_train_params(&params, &generator, WINW, WINH, s);

        printf("RECALL = %f, PRECSION = %f, DEPTH = %d\n", params.recall, params.precision, params.depth);

        generator.scs = cc->sc;
        generator.scSize = s;

        int needSize = posSet->ssize * generator.npRate;
        printf("GENERATE NEGATIVE SAMPLES: %d\n", needSize);
        generate_negative_samples(negSet, &generator, needSize);
        train(cc->sc + s, posSet, negSet, &params);

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

        release_data(negSet);
        printf("---------------------------------------------\n");
    }

    fclose(generator.fin);

    release(&posSet);
    release(&negSet);

}


int predict(Cascade *cc, uint32_t *iImg, int iStride, float &score){
    score = 0;

    for(int i = 0; i < cc->ssize; i++){
        float t;
        if(predict(cc->sc + i, iImg, iStride, t) == 0)
            return 0;

        score = t;
    }

    return 1;
}


void refine_samples(Cascade *cc, SampleSet *set, int flag){
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



void init_detect_factor(Cascade *cc, float startScale, float endScale, float offset, int layer){
    cc->startScale = startScale;
    cc->endScale = endScale;
    cc->layer = layer;
    cc->offsetFactor = offset;

    float stepFactor = powf(endScale / startScale, 1.0f / layer);
    printf("stepFactor: %f\n", stepFactor);
}


//#define MERGE_RECT

int merge_rect(HRect *rects, float *scores, int &size){
    if(size < 2) return size;

    int8_t *flags = new int8_t[size]; assert(flags != NULL);

    memset(flags, 0, sizeof(int8_t) * size);

    for(int i = 0; i < size; i++){
        int xi0 = rects[i].x;
        int yi0 = rects[i].y;
        int xi1 = rects[i].x + rects[i].width - 1;
        int yi1 = rects[i].y + rects[i].height - 1;

        int cix = (xi0 + xi1) >> 1;
        int ciy = (yi0 + yi1) >> 1;
        int sqi = rects[i].width * rects[i].height;

        for(int j = i + 1; j < size; j++){
            int xj0 = rects[j].x;
            int yj0 = rects[j].y;
            int xj1 = rects[j].x + rects[j].width - 1;
            int yj1 = rects[j].y + rects[j].height - 1;

            int cjx = (xj0 + xj1) >> 1;
            int cjy = (yj0 + yj1) >> 1;

            int sqj = rects[j].width * rects[j].height;

            bool acInB = (xi0 <= cjx && cjx <= xi1) && (yi0 <= cjy && cjy <= yi1);
            bool bcInA = (xj0 <= cix && cix <= xj1) && (yj0 <= ciy && ciy <= yj1);
            bool acNInB = (cjx < xi0 || cjx > xi1) || (cjy < yi0 || cjy > yi1);
            bool bcNInA = (cix < xj0 || cix > xj1) || (ciy < yj0 || ciy > yj1);

            if(acInB && bcInA){
                if(scores[j] > scores[i])
                    flags[i] = 1;
                else
                    flags[j] = 1;
            }
            else if(acInB && bcNInA){
                 flags[j] = 1;
            }
            else if(acNInB && bcInA){
                flags[i] = 1;
            }
        }
    }

    for(int i = 0; i < size; i++){
        if(flags[i] == 0) continue;

        flags[i] = flags[size - 1];
        rects[i] = rects[size - 1];
        scores[i] = scores[size - 1];
        size --;
        i--;
    }

    delete []flags;
    flags = NULL;

    return size;
}


#define FIX_Q 14

int detect_one_scale(Cascade *cc, float scale, uint32_t *iImg, int width, int height, int stride, HRect *resRect, float *resScores){
    int WINW = cc->WINW;
    int WINH = cc->WINH;

    int dx = WINW * cc->offsetFactor;
    int dy = WINH * cc->offsetFactor;

    int count = 0;
    float score;

    int HALF_ONE = 1 << (FIX_Q - 1);
    int FIX_SCALE = scale * (1 << FIX_Q);

    int x0 = -1;
    int x1 = WINW - 1;
    int y0 = -stride;
    int y1 = (WINH - 1) * stride;

    for(int y = 0; y <= height - WINH; y += dy){
        for(int x = 0; x <= width - WINW; x += dx){
            uint32_t *ptr = iImg + y * stride + x;

            if(predict(cc->sc, cc->ssize, iImg + y * stride + x, stride, score) == 1){
                resRect[count].x = (x * FIX_SCALE + HALF_ONE) >> FIX_Q;
                resRect[count].y = (y * FIX_SCALE + HALF_ONE) >> FIX_Q;

                resRect[count].width = (WINW * FIX_SCALE + HALF_ONE) >> FIX_Q;
                resRect[count].height = (WINH * FIX_SCALE + HALF_ONE) >> FIX_Q;

                resScores[count] = score;
                count++;
            }
        }
    }

#ifdef MERGE_RECT
    count = merge_rect(resRect, resScores, count);
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

int detect(Cascade *cc, uint8_t *img, int width, int height, int stride, HRect **resRect){
    int WINW, WINH, capacity;
    float scale, stepFactor;

    uint8_t *dImg, *ptrSrc, *ptrDst;
    uint32_t *iImgBuf, *iImg;

    HRect *rects;
    float *scores;
    int top = 0;

    int srcw, srch, srcs, dstw, dsth, dsts;
    int minSide;
    int count;

    WINW = cc->WINW;
    WINH = cc->WINH;

    scale = cc->startScale;
    stepFactor = powf(cc->endScale / cc->startScale, 1.0f / (cc->layer));

    capacity = calculate_max_size(width, height, scale, HU_MAX(WINW, WINH));

    dImg = new uint8_t[capacity * 2]; assert(dImg != NULL);
    iImgBuf = new uint32_t[capacity * 2]; assert(iImgBuf != NULL);

    const int BUFFER_SIZE = 10000;
    rects  = new HRect[BUFFER_SIZE]; assert(rects != NULL);
    scores = new float[BUFFER_SIZE]; assert(scores != NULL);

    memset(rects, 0, sizeof(HRect)  * BUFFER_SIZE);
    memset(scores, 0, sizeof(float) * BUFFER_SIZE);

    ptrSrc = img;
    ptrDst = dImg;

    srcw = width;
    srch = height;
    srcs = stride;

    count = 0;

    minSide = HU_MIN(width, height);

    for(int i = 0; i < cc->layer; i++){
        float scale2 = HU_MIN(WINW, WINH) / (minSide * scale);

        dstw = scale2 * width;
        dsth = scale2 * height;
        dsts = dstw;

        resizer_bilinear_gray(ptrSrc, srcw, srch, srcs, ptrDst, dstw, dsth, dsts);

        /*
        //cv::Mat sImg(srcw, srch, CV_8UC1, ptrSrc, srcs);
        cv::Mat sImg(dsth, dstw, CV_8UC1, ptrDst, dsts);
        cv::imshow("sImg", sImg);
        cv::waitKey();
        //*/

        assert(dstw * dsth < 16843009);

        memset(iImgBuf, 0, sizeof(uint32_t) * (dstw + 1) * (dsth + 1));
        iImg = iImgBuf + dstw + 1 + 1;

        integral_image(ptrDst, dstw, dsth, dsts, iImg, dstw + 1);

        count += detect_one_scale(cc, 1.0f / scale2, iImg, dstw, dsth, dstw + 1, rects + count, scores + count);

        ptrSrc = ptrDst;

        srcw = dstw;
        srch = dsth;
        srcs = dsts;

        if(ptrDst == dImg)
            ptrDst = dImg + dstw * dsth;
        else
            ptrDst = dImg;

        scale *= stepFactor;
    }

    if(count > 0){
#ifdef MERGE_RECT
        count = merge_rect(rects, scores, count);
#endif

        *resRect = new HRect[count]; assert(resRect != NULL);
        memcpy(*resRect, rects, sizeof(HRect) * count);
    }

    delete [] dImg;
    delete [] iImgBuf;
    delete [] rects;
    delete [] scores;

    return count;
}


int load(Cascade *cc, const char *filePath){
    if(cc == NULL)
        return 1;

    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 2;
    }

    int ret;

    ret = fread(&cc->ssize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&cc->WINW, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&cc->WINH, sizeof(int), 1, fin); assert(ret == 1);

    cc->sc = new StrongClassifier[cc->ssize]; assert(cc->sc != NULL);

    memset(cc->sc, 0, sizeof(StrongClassifier) * cc->ssize);

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


int save(Cascade *cc, const char *filePath){
    FILE *fout = fopen(filePath, "wb");
    if(fout == NULL)
        return 1;

    int ret;

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


void release_data(Cascade *cc){
    if(cc->sc != NULL)
        return;

    for(int i = 0; i < cc->ssize; i++)
        release_data(cc->sc);

    delete [] cc->sc;
    cc->sc = NULL;
}


void release(Cascade **cc){
    if(*cc == NULL)
        return;

    release_data(*cc);
    delete cc;
    cc = NULL;
}


