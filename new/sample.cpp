#include "sample.h"


void release_data(Sample *sample){
    if(sample->img != NULL)
        delete [] sample->img;
    sample->img = NULL;

    if(sample->iImgBuf != NULL)
        delete [] sample->iImgBuf;
    sample->iImgBuf = NULL;

    sample->iImg = NULL;
}


void release(Sample **sample){
    if(*sample != NULL){
        release_data(*sample);
        delete *sample;
    }

    *sample = NULL;
}


int read_samples(const char *fileList, int ssize, int WINW, int WINH, int mirrorFlag, SampleSet **posSet){
    std::vector<std::string> imgList;
    char rootDir[256], fileName[256], ext[30];
    int ret;

    if(load("pos_samples.bin", posSet) == 0){
        printf("POSITIVE SAMPLE SIZE: %d\n", (*posSet)->ssize);
        return (*posSet)->ssize;
    }

    if(read_file_list(fileList, imgList) != 0)
        return 0;

    if(ssize <= 100)
        ssize = imgList.size();
    else
        ssize = HU_MIN(ssize, imgList.size());

    if(ssize <= 0) return 0;


    *posSet = new SampleSet;

    (*posSet)->winw = WINW;
    (*posSet)->winh = WINH;

    if(mirrorFlag == 1)
        (*posSet)->ssize = ssize * 2;
    else
        (*posSet)->ssize = ssize;

    (*posSet)->samples = new Sample*[(*posSet)->ssize];

    memset((*posSet)->samples, 0, sizeof(Sample*) * (*posSet)->ssize);

    for(int i = 0; i < ssize; i++){
        const char *imgPath = imgList[i].c_str();

        cv::Mat img = cv::imread(imgPath, 0);
        assert(!img.empty());

        analysis_file_path(imgPath, rootDir, fileName, ext);

        Sample *sample = new Sample;

        sample->img = new uint8_t[WINW * WINH];
        sample->iImgBuf = new uint32_t[(WINW + 1) * (WINH + 1)];
        sample->iImg = sample->iImgBuf + (WINW + 1) + 1;
        sample->istride = WINW + 1;
        sample->stride = WINW;

        sprintf(sample->patchName, "%s", fileName);

        resizer_bilinear_gray(img.data, img.cols, img.rows, img.step,
                sample->img, WINW, WINH, WINW);

        memset(sample->iImgBuf, 0, sizeof(uint32_t) * (WINW + 1) * (WINH + 1));

        integral_image(sample->img, WINW, WINH, WINW, sample->iImg, WINW + 1);

        if(mirrorFlag == 0){
            (*posSet)->samples[i] = sample;
        }
        else {
            (*posSet)->samples[i * 2] = sample;

            //horizontal
            sample = new Sample;

            sample->img = new uint8_t[WINW * WINH];
            sample->iImgBuf = new uint32_t[(WINW + 1) * (WINH + 1)];
            sample->iImg = sample->iImgBuf + (WINW + 1) + 1;
            sample->istride = WINW + 1;
            sample->stride = WINW;

            sprintf(sample->patchName, "%s_h", fileName);

            memcpy(sample->img, (*posSet)->samples[i * 2]->img, sizeof(uint8_t) * WINW * WINH);

            horizontal_mirror(sample->img, WINW, WINH, WINW);

            memset(sample->iImgBuf, 0, sizeof(uint32_t) * (WINW + 1) * (WINH + 1));

            integral_image(sample->img, WINW, WINH, WINW, sample->iImg, WINW + 1);


            (*posSet)->samples[i * 2 + 1] = sample;
        }

        printf("%d\r", i + 1); fflush(stdout);
    }

    (*posSet)->capacity = (*posSet)->ssize;

    save("pos_samples.bin", *posSet);

    return ssize;
}


int save(const char *filePath, SampleSet *set){
    FILE *fout = fopen(filePath, "wb");

    if(fout == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;

    int WINW = set->winw;
    int WINH = set->winh;

    ret = fwrite(&set->ssize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->winw, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->winh, sizeof(int), 1, fout); assert(ret == 1);

    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];

        ret = fwrite(sample->img, sizeof(uint8_t), WINW * WINH, fout); assert(ret == WINW * WINH);
        ret = fwrite(sample->patchName, sizeof(char), 100, fout); assert(ret == 100);
    }

    fclose(fout);

    return 0;
}


int load(const char *filePath, SampleSet **resSet){
    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;
    int WINW, WINH, ssize;

    SampleSet *set = new SampleSet;
    *resSet = set;

    memset(set, 0, sizeof(SampleSet));


    ret = fread(&ssize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->winw, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->winh, sizeof(int), 1, fin); assert(ret == 1);

    printf("%d %d %d\n", ssize, set->winw, set->winh);

    WINW = set->winw;
    WINH = set->winh;

    reserve(set, ssize);

    for(int i = 0; i < ssize; i++){
        Sample *sample = new Sample;
        set->samples[i] = sample;

        memset(sample, 0, sizeof(Sample));

        sample->img = new uint8_t[WINW * WINH];
        sample->iImgBuf = new uint32_t[(WINW + 1) * (WINH + 1)];
        sample->iImg = sample->iImgBuf + WINW + 1 + 1;

        sample->stride = WINW;
        sample->istride = WINW + 1;

        sample->score = 0;

        ret = fread(sample->img, sizeof(uint8_t), WINW * WINH, fin); assert(ret == WINW * WINH);
        ret = fread(sample->patchName, sizeof(char), 100, fin); assert(ret == 100);

        integral_image(sample->img, WINW, WINH, WINW, sample->iImg, sample->istride);
    }

    set->ssize = ssize;

    fclose(fin);

    return 0;
}



void add_sample(SampleSet *set, Sample *sample){
    if(set->ssize == set->capacity){
        if(set->ssize > 0){
            Sample **samples = new Sample*[set->ssize * 2];

            memcpy(samples, set->samples, sizeof(Sample*) * set->ssize);
            memset(samples + set->ssize, 0, sizeof(Sample*) * set->ssize);
            memset(set->samples, 0, sizeof(Sample*) * set->ssize);

            delete [] set->samples;

            set->samples = samples;
            set->capacity = set->ssize * 2;
        }
        else {
            set->samples = new Sample*[100];
            memset(set->samples, 0, sizeof(Sample*) * 100);
            set->capacity = 100;
        }
    }

    set->samples[set->ssize] = sample;
    set->ssize ++;
}


void add_sample_capacity_unchange(SampleSet *set, Sample *sample){
    if(set->capacity <= set->ssize){
        cv::RNG rng(cv::getTickCount());
        int len = 0.1 * set->ssize + 1;

        for(int i = 0; i < len; i++){
            int id = rng.uniform(0, set->ssize);

            HU_SWAP(set->samples[id], set->samples[set->ssize - 1], Sample*);
            release(&set->samples[set->ssize - 1]);

            set->ssize --;
        }
    }

    set->samples[set->ssize++] = sample;
}
//扩大SampleSet元素空间
void reserve(SampleSet *set, int size){
    if(set->capacity > size)
        return ;

    Sample **samples = new Sample*[size];

    memset(samples, 0, sizeof(Sample*) * size);

    if(set->ssize > 0){
        memcpy(samples, set->samples, sizeof(Sample*) * set->ssize);
        memset(set->samples, 0, sizeof(Sample*) * set->ssize);
    }

    if(set->samples != NULL)
        delete [] set->samples;

    set->samples = samples;
    set->capacity = size;
}

#define WIN_SIZE 64
#define LEN 8
#define DSIZE 8

uint64_t generate_hash_key(uint8_t *img, int winw, int winh, int stride){
    int lenx = winw / DSIZE;
    int leny = winh / DSIZE;

    uint32_t sum[64];
    int dsize2 = DSIZE * DSIZE;

    memset(sum, 0, sizeof(uint32_t) * 64);

    for(int y = 0; y < winh; y ++){
        int iy = y / leny * DSIZE;

        for(int x = 0; x < winw; x++){
            int ix = x / lenx;

            sum[iy + ix] += img[y * stride + x];
        }
    }

    uint32_t mean = 0;

    for(int i = 0; i < dsize2; i++)
        mean += sum[i];

    mean /= dsize2;


    uint64_t code = 0;

    for(int i = 0; i < dsize2; i++){
        code <<= 1;
        code |= (sum[i] > mean);
    }

    return code;
}


typedef struct {
    int id;
    uint64_t code;
} HashPair;

#define LT_PAIR(a, b) ((a).code < (b).code)

IMPLEMENT_QSORT(sort_arr_pair, HashPair, LT_PAIR);

void uniq(SampleSet *set){
    HashPair *pairs = new HashPair[set->ssize];

    assert(set->winw == WIN_SIZE && set->winh == WIN_SIZE);

    int stride = set->winw;
    for(int i = 0; i < set->ssize; i++){
        pairs[i].id = i;
        pairs[i].code = generate_hash_key(set->samples[i]->img, set->winw, set->winh, stride);
    }

    sort_arr_pair(pairs, set->ssize);

    uint64_t lastCode = pairs[0].code;
    for(int i = 1; i < set->ssize; i++){
        if(lastCode == pairs[i].code)
            release(set->samples + pairs[i].id);
        else
            lastCode = pairs[i].code;
    }

    for(int i = 0; i < set->ssize; i++){
        if(set->samples[i] == NULL){
            HU_SWAP(set->samples[i], set->samples[set->ssize - 1], Sample*);
            set->ssize --;
            i --;
        }
    }

    delete [] pairs;
}


void merge(SampleSet *sets, int size, SampleSet *res, int count){
    int ssize = 0;

    for(int i = 0; i < size; i++)
        ssize += sets[i].ssize;

    release_data(res);

    res->winw = sets[0].winw;
    res->winh = sets[0].winh;
    res->samples = new Sample*[count];
    res->capacity = count;

    memset(res->samples, 0, sizeof(Sample*) * count);

    if(ssize > count){
        float *scores = new float[ssize];
        int idx = 0;

        for(int i = 0; i < size; i++){
            for(int j = 0; j < sets[i].ssize; j++)
                scores[idx ++] = sets[i].samples[j]->score;
        }

        assert(idx == ssize);
        sort_arr_float(scores, ssize);

        float thresh = scores[ssize - count];

        for(int i = 0; i < size; i++){
            SampleSet *set = sets + i;
            for(int j = 0; j < set->ssize; j++){
                if(set->samples[j]->score > thresh){
                    res->samples[res->ssize++] = set->samples[j];
                    set->samples[j] = NULL;
                }
            }

            release_data(set);
        }

        assert(res->ssize <= res->capacity);
        delete [] scores;
    }
    else {
        for(int i = 0; i < size; i++){
            SampleSet *set = sets + i;
            for(int j = 0; j < set->ssize; j++){
                res->samples[res->ssize++] = set->samples[j];
                set->samples[j] = NULL;
            }

            release_data(set);
        }
    }
}


//分割样本集，把一部分样本分配给res，采用随机采样的方式
void split(SampleSet *src, float rate, SampleSet *res){
    cv::RNG rng(cv::getTickCount());
    int count = rate * src->ssize;

    if(count == 0) {
        res->ssize = 0;
        return;
    }

    release_data(res);

    res->winw = src->winw;
    res->winh = src->winh;
    res->ssize = 0;
    res->samples = new Sample*[count];

    for(int i = 0; i < count; i++){
        int id = rng.uniform(0, src->ssize);

        res->samples[i] = src->samples[id];
        src->samples[id] = src->samples[src->ssize - 1];
        src->samples[src->ssize - 1] = NULL;

        src->ssize --;
        res->ssize ++;
    }

    res->capacity = res->ssize;
}



//用给出的图像生成样本
void create_sample(Sample *sample, uint8_t *img, int width, int height, int stride, const char *patchName){
    if(sample->img == NULL)
        sample->img = new uint8_t[width * height];

    if(sample->iImgBuf == NULL)
        sample->iImgBuf = new uint32_t[(width + 1) * (height + 1)];

    sample->iImg = sample->iImgBuf + (width + 1) + 1;
    sample->istride = width + 1;
    sample->stride = width;
    sample->score = 0;

    strcpy(sample->patchName, patchName);

    for(int y = 0; y < height; y++)
        memcpy(sample->img + y * width, img + y * stride, sizeof(uint8_t) * width);

    memset(sample->iImgBuf, 0, sizeof(uint32_t) * (width + 1) * (height + 1));

    integral_image(sample->img, width, height, width, sample->iImg, sample->istride);
}


void insert_sample(SampleSet *set, uint8_t *img, int width, int height, int stride, float score, const char *patchName, float recall){
#ifdef TEST_TIME
    struct timezone tz;
    struct timeval stv, etv;
#endif

    if(set->ssize == set->capacity){
#ifdef TEST_TIME
        gettimeofday(&stv, &tz);
#endif
        float *scores = new float[set->ssize];

        for(int i = 0; i < set->ssize; i++)
            scores[i] = set->samples[i]->score;

        sort_arr_float(scores, set->ssize);

        float thresh = scores[int((1 - recall) * set->ssize)] - 0.000001f;

        for(int i = 0; i < set->ssize; i++){
            if(set->samples[i]->score <= thresh){
                HU_SWAP(set->samples[i], set->samples[set->ssize - 1], Sample*);
                i --;
                set->ssize --;
            }
        }
#ifdef TEST_TIME
        gettimeofday(&etv, &tz);
        printf("%.2f ms\n", (etv.tv_sec - stv.tv_sec) * 1000 + (etv.tv_usec - stv.tv_usec) * 0.0001f);
#endif
        delete [] scores;
    }
    else{
        Sample *sample = set->samples[set->ssize];

        sample->istride = set->winw + 1;
        sample->stride = set->winw;

        assert(sample != NULL && sample->img != NULL && sample->iImgBuf != NULL);

        for(int y = 0; y < set->winh; y++)
            memcpy(sample->img + y * set->winw, img + y * stride, sizeof(uint8_t) * set->winw);

        memset(sample->iImgBuf, 0, sizeof(uint32_t) * (set->winw + 1) * (set->winh + 1));

        integral_image(sample->img, set->winw, set->winh, set->winw, sample->iImg, sample->istride);

        sample->score = score;
        strcpy(sample->patchName, patchName);

        set->ssize++;
    }
}


void random_subset(SampleSet *oriSet, SampleSet *subset, int size){
    cv::RNG rng(cv::getTickCount());
    int *idxs = NULL;

    assert(size > 0 && size <= oriSet->ssize);

    subset->winw = oriSet->winw;
    subset->winh = oriSet->winh;

    reserve(subset, size);

    idxs = new int[size];

    for(int i = 0; i < size; i++){
        idxs[i] = i;
    }

    for(int i = 0; i < size; i++){
        int id1 = rng.uniform(0, i);
        int id2 = rng.uniform(0, i);

        if(id1 != id2)
            HU_SWAP(idxs[id1], idxs[id2], int);
    }

    for(int i = 0; i < size; i++){
        subset->samples[i] = oriSet->samples[idxs[i]];
    }

    subset->ssize = size;

    delete [] idxs;
}


void release_data(SampleSet *set){
    if(set != NULL){
        if(set->samples != NULL){
            for(int i = 0; i < set->capacity; i++){
                if(set->samples[i] != NULL)
                    release(&set->samples[i]);

                set->samples[i] = NULL;
            }

            delete [] set->samples;
        }

        set->samples = NULL;

        set->ssize = 0;
        set->capacity = 0;
    }
}


void release(SampleSet **set){
    release_data(*set);

    delete *set;
    *set = NULL;
}


void print_info(SampleSet *set, const char *tag){
    printf("SampleSet: %s\n", tag);

    printf("ssize: %d, winw: %d, winh: %d\n", set->ssize, set->winw, set->winh);
    printf("capacity: %d\n", set->capacity);
}


void write_images(SampleSet *set, const char *outDir, int step){
    int ssize = set->ssize;

    int WINW = set->winw;
    int WINH = set->winh;

    char outPath[256], command[256];

    sprintf(command, "mkdir -p %s", outDir);

    int ret = system(command);

    for(int i = 0; i < ssize; i+= step){
        Sample *sample = set->samples[i];
        cv::Mat img(WINH, WINW, CV_8UC1, sample->img, sample->stride);

        sprintf(outPath, "%s/%s.jpg", outDir, sample->patchName);
        if(!cv::imwrite(outPath, img))
            printf("Can't write image %s\n", outPath);

        printf("%d\r", i); fflush(stdout);
    }
}
