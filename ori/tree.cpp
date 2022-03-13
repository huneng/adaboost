#include "tree.h"


static void generate_feature_templates(int WINW, int WINH, FeatTemp *temp, int featNum){
    static cv::RNG rng(cv::getTickCount());

    memset(temp, 0, sizeof(FeatTemp) * featNum);

    for(int i = 0; i < featNum; i++){
        FeatTemp ft;
        int x0, y0, x1, y1, w, h;

        while(1){
            x0 = rng.uniform(1, WINW - 2);
            y0 = rng.uniform(1, WINH - 2);
            x1 = rng.uniform(1, WINW - 2);
            y1 = rng.uniform(1, WINH - 2);

            w = rng.uniform(1, WINW - 1);
            h = rng.uniform(1, WINH - 1);

            if(x0 + w >= WINW || y0 + h >= WINH || x1 + w >= WINW || y1 + h >= WINH )
                continue;

            int cx0 = x0 + (w >> 1);
            int cy0 = y0 + (h >> 1);
            int cx1 = x1 + (w >> 1);
            int cy1 = y1 + (h >> 1);

            if(abs(cx0 - cx1) < w && abs(cy1 - cy0) < h){
                if(abs(x0 - x1) < 2 || abs(y0 - y1) < 2)
                    continue;

                int x0_ = HU_MAX(x0, x1);
                int x1_ = HU_MIN(x0 + w - 1, x1 + w - 1);
                int y0_ = HU_MAX(y0, y1);
                int y1_ = HU_MIN(y0 + h - 1, y1 + h - 1);

                if((x1_ - x0_ + 1) * (y1_ - y0_ + 1) >= (w * h >> 2))
                    continue;
            }

            if(x0 > x1){
                HU_SWAP(x0, x1, uint8_t);
                HU_SWAP(y0, y1, uint8_t);
            }

            int flag = 0;

            ft.x00 = x0 - 1;
            ft.y00 = y0 - 1;
            ft.x01 = x0 + w - 1;
            ft.y01 = y0 + h - 1;

            ft.x10 = x1 - 1;
            ft.y10 = y1 - 1;
            ft.x11 = x1 + w - 1;
            ft.y11 = y1 + h - 1;

            for(int j = 0; j < i - 1; j++){
                if(memcmp(temp + j, &ft, sizeof(FeatTemp)) == 0){
                    flag = 1;
                    break;
                }
            }

            if(flag) continue;

            break;
        }

        temp[i] = ft;
    }
}


#define GET_VALUE(ft, img, stride, value) \
{ \
    int x0, y0, x1, y1;                   \
    int a, b;                             \
    x0 = ft->x00;                         \
    y0 = ft->y00 * stride;                \
    x1 = ft->x01;                         \
    y1 = ft->y01 * stride;                \
                                          \
    a = img[y1 + x1] - img[y1 + x0] - img[y0 + x1] + img[y0 + x0]; \
                                          \
    x0 = ft->x10;                         \
    y0 = ft->y10 * stride;                \
    x1 = ft->x11;                         \
    y1 = ft->y11 * stride;                \
                                          \
    b = img[y1 + x1] - img[y1 + x0] - img[y0 + x1] + img[y0 + x0]; \
    value = (a - b); \
}


static float get_value(FeatTemp *ft, uint32_t *img, int stride){
    int x0, y0, x1, y1;
    int a, b;

    x0 = ft->x00;
    y0 = ft->y00 * stride;
    x1 = ft->x01;
    y1 = ft->y01 * stride;

    a = img[y1 + x1] - img[y1 + x0] - img[y0 + x1] + img[y0 + x0];

    x0 = ft->x10;
    y0 = ft->y10 * stride;
    x1 = ft->x11;
    y1 = ft->y11 * stride;

    b = img[y1 + x1] - img[y1 + x0] - img[y0 + x1] + img[y0 + x0];

    return (a - b);
}


static void extract_features(SampleSet *set, int *idxs, int ssize, FeatTemp *featTemps, int fsize, float *feats){

    memset(feats, 0, sizeof(float) * fsize * ssize);

    for(int i = 0; i < ssize; i++){
        int j = idxs[i];
        Sample *sample = set->samples[j];
        uint32_t *iImg = sample->iImg;
        int istride =  sample->istride;

        float *ptrFeats = feats + i; //the ith sample

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
        for(int f = 0; f < fsize; f++){
            ptrFeats[f * ssize] = get_value(featTemps + f, iImg, istride);
        }
    }
}


#define VALUE_LENGTH 1024

static int statistic(float *feats, double *weights, int size, float minValue, float rstep,  double *weightTable, int *countTable){
    double sumw = 0.0;
    int nonezero = 0;

    memset(weightTable, 0, sizeof(double) * VALUE_LENGTH);
    memset(countTable, 0, sizeof(int) * VALUE_LENGTH);

    for(int i = 0; i < size; i++){
        int id = (feats[i] - minValue) * rstep;

        weightTable[id] += weights[i];
        countTable[id] ++;

        sumw += weights[i];
    }

    sumw = 1.0 / sumw;

    weightTable[0] *= sumw;
    for(int i = 1; i < VALUE_LENGTH; i++){
        weightTable[i] *= sumw;
        weightTable[i] += weightTable[i - 1];

        nonezero += (countTable[i] != 0);
        countTable[i] += countTable[i - 1];
    }

    return nonezero;
}


static void train_weak_classifier(float *posFeats, double *posWeights, int posSize,
        float *negFeats, double *negWeights, int negSize, float rate, float &thresh, double &error){
    double posWT[VALUE_LENGTH], negWT[VALUE_LENGTH];

    int pcount[VALUE_LENGTH], ncount[VALUE_LENGTH];

    float maxValue, minValue;
    float featStep, rfeatStep;

    maxValue = -FLT_MAX, minValue = FLT_MAX;

    for(int i = 0; i < posSize; i++){
        maxValue = HU_MAX(maxValue, posFeats[i]);
        minValue = HU_MIN(minValue, posFeats[i]);
    }

    for(int i = 0; i < negSize; i++){
        maxValue = HU_MAX(maxValue, negFeats[i]);
        minValue = HU_MIN(minValue, negFeats[i]);
    }

    maxValue = HU_MAX(minValue + 1, maxValue);

    featStep = (maxValue - minValue) / (VALUE_LENGTH - 1);
    assert(featStep > 0);

    rfeatStep = 1.0f / featStep;

    if(statistic(posFeats, posWeights, posSize, minValue, rfeatStep, posWT, pcount) < 5) return;
    if(statistic(negFeats, negWeights, negSize, minValue, rfeatStep, negWT, ncount) < 5) return;

    int CLASS_MIN_SIZE_P = rate * posSize;
    int CLASS_MIN_SIZE_N = rate * negSize;

    int i = 0;
    int len = VALUE_LENGTH - 1;

    for(; i < VALUE_LENGTH; i++)
        if(pcount[i + 1] >= CLASS_MIN_SIZE_P && ncount[i + 1] >= CLASS_MIN_SIZE_N)
            break;

    for(; len >= i; len --)
        if(posSize - pcount[len] >= CLASS_MIN_SIZE_P &&
                negSize - ncount[len] >= CLASS_MIN_SIZE_N)
            break;

    for(; i <= len; i++){
        double WL = (posWT[i] + 1.0 - negWT[i]) * 0.5f;
        double WR = 1.0 - WL;

        double e = - WL * log(WL) - WR * log(WR);

        if(e < error){
            error = e;
            thresh = minValue + i * featStep;
        }
    }
}


typedef struct{
    int *posIdxs;
    double *pws;
    int posSize;

    int *negIdxs;
    double *nws;
    int negSize;
} NodePair;


void init_rates(float *rate, int depth){
    assert(depth > 3 && depth < 7);

    if(depth == 4){
        rate[0] = 0.2;

        rate[1] = 0.1;
        rate[2] = 0.1;

        for(int i = 3; i < 7; i++)
            rate[i] = 0.05;
    }
    else if(depth == 5){
        rate[0] = 0.4;

        rate[1] = 0.3;
        rate[2] = 0.3;

        for(int i = 3; i < 7; i++)
            rate[i] = 0.15;

        for(int i = 7; i < 15; i++)
            rate[i] = 0.05;
    }
    else if(depth == 6){
        rate[0] = 0.4;

        rate[1] = 0.3;
        rate[2] = 0.3;

        for(int i = 3; i < 7; i++)
            rate[i] = 0.2;

        for(int i = 7; i < 15; i++)
            rate[i] = 0.1;

        for(int i = 15; i < 31; i ++)
            rate[i] = 0.05;
    }
}


void create(NodePair *pair, int posSize, int negSize){
    assert(posSize > 0 && negSize > 0);

    pair->posSize = posSize;
    pair->negSize = negSize;

    pair->posIdxs = new int[posSize];
    pair->pws = new double[posSize];

    pair->negIdxs = new int[negSize];
    pair->nws = new double[negSize];
}


void release_data(NodePair *pair){
    if(pair->posIdxs != NULL)
        delete [] pair->posIdxs;
    pair->posIdxs = NULL;

    if(pair->pws != NULL)
        delete [] pair->pws;
    pair->pws = NULL;

    if(pair->negIdxs != NULL)
        delete [] pair->negIdxs;
    pair->negIdxs = NULL;

    if(pair->nws != NULL)
        delete [] pair->nws;
    pair->nws = NULL;

    pair->negSize = 0;
    pair->posSize = 0;
}


static void split(float *posFeats, float *negFeats, float thresh, NodePair *ppair, NodePair *lpair, NodePair *rpair){
    lpair->posSize = 0;
    lpair->negSize = 0;

    rpair->posSize = 0;
    rpair->negSize = 0;

    for(int i = 0; i < ppair->posSize; i++){
        if(posFeats[i] <= thresh){
            lpair->posIdxs[lpair->posSize] = ppair->posIdxs[i];
            lpair->pws[lpair->posSize] = ppair->pws[i];

            lpair->posSize++;
        }
        else {
            rpair->posIdxs[rpair->posSize] = ppair->posIdxs[i];
            rpair->pws[rpair->posSize] = ppair->pws[i];

            rpair->posSize++;
        }
    }

    for(int i = 0; i < ppair->negSize; i++){
        if(negFeats[i] <= thresh){
            lpair->negIdxs[lpair->negSize] = ppair->negIdxs[i];
            lpair->nws[lpair->negSize] = ppair->nws[i];

            lpair->negSize++;
        }
        else {
            rpair->negIdxs[rpair->negSize] = ppair->negIdxs[i];
            rpair->nws[rpair->negSize] = ppair->nws[i];

            rpair->negSize++;
        }
    }

    int minLeafSize = 20;

    assert(lpair->posSize > minLeafSize && lpair->negSize > minLeafSize);
    assert(rpair->posSize > minLeafSize && rpair->negSize > minLeafSize);
}

#define FEAT_NUM 500

float train(Tree* root, int depth, SampleSet *posSet, SampleSet *negSet, float recall, int featTimes){

    int nlSize   = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);

    int nodeSize = nlSize - leafSize;

    int posSize = posSet->ssize * 0.9;
    int negSize = negSet->ssize * 0.9;

    FeatTemp *sfts = new FeatTemp[FEAT_NUM * featTimes];

    float *posFeats = new float[FEAT_NUM * posSize];
    float *negFeats = new float[FEAT_NUM * negSize];

    NodePair *pairs = new NodePair[nlSize];

    float *bestPosFeats = new float[posSize];
    float *bestNegFeats = new float[negSize];

    float rates[32];

    memset(pairs, 0, sizeof(NodePair) * nlSize);

    random_order(posSet);
    random_order(negSet);

    //create root pair and set values
    create(pairs, posSize, negSize);

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < posSize; i++){
        pairs[0].posIdxs[i] = i;
        pairs[0].pws[i] = exp(-posSet->samples[i]->score);
    }
    update_weights(pairs[0].pws, posSize);

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < negSize; i++){
        pairs[0].negIdxs[i] = i;
        pairs[0].nws[i] = exp(negSet->samples[i]->score);
    }
    update_weights(pairs[0].nws, negSize);

    //init rate for tree binary split
    init_rates(rates, depth);

    for(int i = 0; i < nodeSize; i++){
        Node *node = root + i;

        int idL = i * 2 + 1;
        int idR = i * 2 + 2;

        float rate = rates[i];
        int *posIdxs, *negIdxs;
        double *tpws, *tnws;

        double sumpw = 0.0f, sumnw = 0.0f;

        posIdxs = pairs[i].posIdxs;
        negIdxs = pairs[i].negIdxs;

        posSize = pairs[i].posSize;
        negSize = pairs[i].negSize;

        tpws = pairs[i].pws;
        tnws = pairs[i].nws;

        for(int j = 0; j < posSize; j++)
            sumpw += tpws[j];

        for(int j = 0; j < negSize; j++)
            sumnw += tnws[j];

        node->pw = sumpw;
        node->nw = sumnw;
        node->posSize = posSize;
        node->negSize = negSize;


        FeatTemp bestFt;
        float bestError = FLT_MAX;
        float bestThresh = 0;

        generate_feature_templates(posSet->winw, posSet->winh, sfts, FEAT_NUM * featTimes);

        for(int k = 0; k < featTimes; k++){

            extract_features(posSet, posIdxs, posSize, sfts + k * FEAT_NUM, FEAT_NUM, posFeats);
            extract_features(negSet, negIdxs, negSize, sfts + k * FEAT_NUM, FEAT_NUM, negFeats);

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
            for(int j = 0; j < FEAT_NUM; j++){
                double error = FLT_MAX;
                float thresh;

                train_weak_classifier(posFeats + j * posSize, tpws, posSize, negFeats + j * negSize, tnws, negSize, rate, thresh, error);

#pragma omp critical
                {
                    if(error < bestError){
                        bestError = error;
                        bestThresh = thresh;

                        memcpy(&bestFt, sfts + j + k * FEAT_NUM, sizeof(FeatTemp));
                        memcpy(bestPosFeats, posFeats + j * posSize, sizeof(float) * posSize);
                        memcpy(bestNegFeats, negFeats + j * negSize, sizeof(float) * negSize);
                    }
                }
            }
        }

        assert(bestError != FLT_MAX);

        node->ft = bestFt;
        node->thresh = bestThresh;
        node->score = 0;

        //print_feature_template(&node->ft, stdout);

        node->lchild = root + idL;
        node->rchild = root + idR;

        create(pairs + idL, posSize, negSize);
        create(pairs + idR, posSize, negSize);

        split(bestPosFeats, bestNegFeats, bestThresh, pairs + i, pairs + idL, pairs + idR);

        release_data(pairs + i);
    }


    for(int i = nodeSize; i < nlSize; i++){
        Node *leaf = root + i;
        NodePair *pair = pairs + i;

        assert(pair->posSize > 0 && pair->negSize > 0);
        double sumpw = 0.0, sumnw = 0.0;

        for(int j = 0; j < pair->posSize; j++)
            sumpw += pair->pws[j];

        for(int j = 0; j < pair->negSize; j++)
            sumnw += pair->nws[j];

        leaf->lchild = NULL;
        leaf->rchild = NULL;

        leaf->score = log((sumpw + 0.000001) / (sumnw + 0.000001)) * 0.5f;

        leaf->pw = sumpw;
        leaf->nw = sumnw;
        leaf->posSize = pair->posSize;
        leaf->negSize = pair->negSize;

        release_data(pair);
    }

    delete [] pairs;
    delete [] sfts;
    delete [] bestPosFeats;
    delete [] bestNegFeats;
    delete [] posFeats;
    delete [] negFeats;


    float *scores = new float[posSet->ssize];

    for(int i = 0; i < posSet->ssize; i++){
        Sample *sample = posSet->samples[i];
        sample->score += predict(root, depth, sample->iImg, sample->istride);

        scores[i] = sample->score;
    }

    quick_sort_float(scores, posSet->ssize);

    float thresh = scores[int((1 - recall) * posSet->ssize)] - 0.000001f;

    printf("thresh: %f , max score: %f\n", thresh, scores[posSet->ssize - 1]);
    delete [] scores;

    for(int i = 0; i < negSet->ssize; i++){
        Sample *sample = negSet->samples[i];
        sample->score += predict(root, depth, sample->iImg, sample->istride);
    }

    return thresh;
}


float predict(Tree *root, int depth, uint32_t *iImg, int istride){
    assert(depth == 4);
#if 0
    for(int i = 0; i < depth - 1; i++){
        float feat;
        FeatTemp *ptrFeat = &root->ft;
        GET_VALUE(ptrFeat, iImg, istride, feat);
        root = root->lchild + (feat > root->thresh);
    }

#else
    //depth 4
    float feat;
    FeatTemp *ptrFeat;

    ptrFeat = &root->ft;
    GET_VALUE(ptrFeat, iImg, istride, feat);
    root = root->lchild + (feat > root->thresh);

    ptrFeat = &root->ft;
    GET_VALUE(ptrFeat, iImg, istride, feat);
    root = root->lchild + (feat > root->thresh);

    ptrFeat = &root->ft;
    GET_VALUE(ptrFeat, iImg, istride, feat);
    root = root->lchild + (feat > root->thresh);

#endif

    return root->score;
}


void print_tree(Tree* root, int depth, FILE *fout){
    static int treeID = 0;
    int nlSize = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);
    int nodeSize = nlSize - leafSize;

    fprintf(fout, "Tree: %d\n", treeID++);
    for(int i = 0; i < nodeSize; i++){
        Node *node = root + i;

        fprintf(fout, "node ID: %2d ", i);
        fprintf(fout, "pws: %10e, nws: %10e, posSize: %6d, negSize: %6d, ", node->pw, node->nw, node->posSize, node->negSize);
        fprintf(fout, "left: %d, right: %d\n", i * 2 + 1, i * 2 + 2);
    }

    for(int i = nodeSize; i < nlSize; i++){
        Node *leaf = root + i;

        fprintf(fout, "node ID: %2d ", i);
        fprintf(fout, "pws: %10e, nws: %10e, posSize: %6d, negSize: %6d, score: %f\n", leaf->pw, leaf->nw, leaf->posSize, leaf->negSize, leaf->score);
    }

    fprintf(fout, "\n");
    fflush(fout);
}


void save(Tree* root, int depth, FILE *fout){
    assert(root != NULL && fout != NULL);

    int nlSize = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);
    int nodeSize = nlSize - leafSize;

    for(int i = 0; i < nlSize; i++){
        Node *node = root + i;

        if(i < nodeSize){
            fwrite(&node->thresh, sizeof(float), 1, fout);
            fwrite(&node->ft, sizeof(FeatTemp), 1, fout);
        }
        else {
            fwrite(&node->score, sizeof(float), 1, fout);
        }
    }
}


void load(Tree* root, int depth, FILE *fin){

    assert(root != NULL && fin != NULL);

    int nlSize = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);
    int nodeSize = nlSize - leafSize;
    int ret;

    for(int i = 0; i < nlSize; i++){
        Node *node = root + i;

        if(i < nodeSize){
            ret = fread(&node->thresh, sizeof(float), 1, fin); assert(ret == 1);
            ret = fread(&node->ft, sizeof(FeatTemp), 1, fin); assert(ret == 1);

            node->lchild = root + i * 2 + 1;
            node->rchild = root + i * 2 + 2;
        }
        else{
            ret = fread(&node->score, sizeof(float), 1, fin); assert(ret == 1);
            node->lchild = NULL;
            node->rchild = NULL;
        }
    }
}


