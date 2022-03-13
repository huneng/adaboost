#include "tree.h"


static void generate_feature_templates(int WINW, int WINH, FeatTemp *temp, int featNum){
    static cv::RNG rng(cv::getTickCount());

    memset(temp, 0, sizeof(FeatTemp) * featNum);

    for(int i = 0; i < featNum; i++){
        FeatTemp ft;

        while(1){
            ft.x0 = rng.uniform(0, WINW);
            ft.y0 = rng.uniform(0, WINH);
            ft.x1 = rng.uniform(0, WINW);
            ft.y1 = rng.uniform(0, WINH);

            ft.w = rng.uniform(2, WINW);
            ft.h = rng.uniform(2, WINH);

            if(ft.x0 + ft.w > WINW ||
                ft.y0 + ft.h > WINH ||
                ft.x1 + ft.w > WINW ||
                ft.y1 + ft.h > WINH )
                continue;

            int cx0 = ft.x0 + (ft.w >> 1);
            int cy0 = ft.y0 + (ft.h >> 1);
            int cx1 = ft.x1 + (ft.w >> 1);
            int cy1 = ft.y1 + (ft.h >> 1);

            if(abs(cx0 - cx1) < ft.w && abs(cy1 - cy0) < ft.h){
                if(abs(ft.x0 - ft.x1) < 2 || abs(ft.y0 - ft.y1) < 2)
                    continue;

                int x0 = HU_MAX(ft.x0, ft.x1);
                int x1 = HU_MIN(ft.x0 + ft.w - 1, ft.x1 + ft.w - 1);
                int y0 = HU_MAX(ft.y0, ft.y1);
                int y1 = HU_MIN(ft.y0 + ft.h - 1, ft.y1 + ft.h - 1);

                if((x1 - x0 + 1) * (y1 - y0 + 1) >= (ft.w * ft.h >> 2))
                    continue;
            }

            if(ft.x0 > ft.x1){
                HU_SWAP(ft.x0, ft.x1, uint8_t);
                HU_SWAP(ft.y0, ft.y1, uint8_t);
            }

            int flag = 0;
            for(int j = 0; j < i; j++){
                if(ft.x0 == temp[j].x0 && ft.y0 == temp[j].y0 &&
                        ft.x1 == temp[j].x1 && ft.y1 == temp[j].y1 &&
                        ft.w == temp[j].w && ft.h == temp[j].h){
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
    x0 = ft->x0 - 1;                      \
    y0 = (ft->y0 - 1) * stride;           \
    x1 = ft->x0 + ft->w - 1;              \
    y1 = (ft->y0 + ft->h - 1) * stride;   \
                                          \
    a = img[y1 + x1] - img[y1 + x0] - img[y0 + x1] + img[y0 + x0]; \
                                          \
    x0 = ft->x1 - 1;                      \
    y0 = (ft->y1 - 1) * stride;           \
    x1 = ft->x1 + ft->w - 1;              \
    y1 = (ft->y1 + ft->h - 1) * stride;   \
                                          \
    b = img[y1 + x1] - img[y1 + x0] - img[y0 + x1] + img[y0 + x0]; \
    value = (a - b); \
}


static float get_value(FeatTemp *ft, uint32_t *img, int stride){
    int x0, y0, x1, y1;
    int a, b;

    x0 = ft->x0 - 1;
    y0 = (ft->y0 - 1) * stride;
    x1 = ft->x0 + ft->w - 1;
    y1 = (ft->y0 + ft->h - 1) * stride;

    a = img[y1 + x1] - img[y1 + x0] - img[y0 + x1] + img[y0 + x0];

    x0 = ft->x1 - 1;
    y0 = (ft->y1 - 1) * stride;
    x1 = ft->x1 + ft->w - 1;
    y1 = (ft->y1 + ft->h - 1) * stride;

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


#define USE_ENTROPY
#define LENGTH 2048

static void binary_classify_error(float *posFeats, double *posWeights, int posSize,
        float *negFeats, double *negWeights, int negSize, float rate, float &thresh, double &error){
    double posWT[LENGTH], negWT[LENGTH];
    double sumPW, sumNW, rsumPW, rsumNW;

    int pcount[LENGTH], ncount[LENGTH];

    float maxFeatValue, minFeatValue;
    float featStep, rfeatStep;

    int num1 = 0, num2 = 0;
    int i;

    memset(posWT, 0, sizeof(double) * LENGTH);
    memset(negWT, 0, sizeof(double) * LENGTH);
    memset(pcount, 0, sizeof(int) * LENGTH);
    memset(ncount, 0, sizeof(int) * LENGTH);

    maxFeatValue = -FLT_MAX, minFeatValue = FLT_MAX;

    for(int i = 0; i < posSize; i++){
        maxFeatValue = HU_MAX(maxFeatValue, posFeats[i]);
        minFeatValue = HU_MIN(minFeatValue, posFeats[i]);
    }

    for(int i = 0; i < negSize; i++){
        maxFeatValue = HU_MAX(maxFeatValue, negFeats[i]);
        minFeatValue = HU_MIN(minFeatValue, negFeats[i]);
    }

    maxFeatValue = HU_MAX(minFeatValue + 1, maxFeatValue);

    featStep = (maxFeatValue - minFeatValue) / (LENGTH - 1);
    assert(featStep > 0);

    rfeatStep = 1.0f / featStep;

    sumPW = 0.0f;
    for(i = 0; i < posSize; i++){
        int idx = (posFeats[i] - minFeatValue) * rfeatStep;
        posWT[idx] += posWeights[i];
        sumPW += posWeights[i];

        pcount[idx] ++;
    }

    rsumPW = 1.0 / sumPW;

    posWT[0] *= rsumPW;
    for(i = 1; i < LENGTH; i++){
        posWT[i] *= rsumPW;
        posWT[i] += posWT[i - 1];

        num1 += (pcount[i] != 0);
        pcount[i] += pcount[i - 1];
    }

    sumNW = 0.0f;
    for(i = 0; i < negSize; i++){
        int idx = (negFeats[i] - minFeatValue) * rfeatStep;
        negWT[idx] += negWeights[i];
        sumNW += negWeights[i];

        ncount[idx] ++;
    }

    rsumNW = 1.0 / sumNW;

    negWT[0] *= rsumNW;
    for(i = 1; i < LENGTH; i++){
        negWT[i] *= rsumNW;
        negWT[i] += negWT[i - 1];

        num2 += (ncount[i] != 0);
        ncount[i] += ncount[i - 1];
    }

    if(num1 <= 5 || num2 <= 5)
        return ;


#ifdef USE_GINI
    double sumW = 2.0;
#endif

    int CLASS_MIN_SIZE_P = rate * posSize;
    int CLASS_MIN_SIZE_N = rate * negSize;

    for(i = 0; i < LENGTH; i++){
        double e;

        if(pcount[i] < CLASS_MIN_SIZE_P || ncount[i] < CLASS_MIN_SIZE_N)
            continue;

        if(posSize - pcount[i] < CLASS_MIN_SIZE_P || negSize - ncount[i] < CLASS_MIN_SIZE_N)
            break;

#if defined(USE_ENTROPY)
        double WL = (posWT[i] + 1 - negWT[i]) / 2;
        double WR = 1 - WL;

        e = - WL * log(WL) - WR * log(WR);

#elif defined(USE_GINI)
        double PWL = posWT[i];
        double NWL = negWT[i];

        double PWR = 1.0 - PWL;
        double NWR = 1.0 - NWL;

        double sumL = PWL + NWL;
        double sumR = PWR + NWR;

        e = ( (sumL / sumW) * (PWL / sumL) * (1 - PWL / sumL) +
                (sumR / sumW) * (PWR / sumR) * (1 - PWR / sumR) );

#else
        e = posWT[i] + (1.0 - negWT[i]);
        e = HU_MIN(e, 2.0 - e);
#endif
        if(e < error){
            error = e;
            thresh = minFeatValue + i * featStep;
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
    assert(depth > 1 && depth < 7);

    if(depth == 2){
        rate[0] = 0.01;
    }
    else if(depth == 3){
        rate[0] = 0.1;
        rate[2] = 0.01;
    }
    else if(depth == 4){
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


#define FEAT_TIMES 4
float train(Tree* root, int depth,
        SampleSet *posSet, float *posFeats, double *pws,
        SampleSet *negSet, float *negFeats, double *nws, int featNum){

    int nlSize   = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);

    int nodeSize = nlSize - leafSize;

    int posSize = posSet->ssize;
    int negSize = negSet->ssize;

    FeatTemp *sfts = new FeatTemp[featNum * FEAT_TIMES];

    NodePair *pairs = new NodePair[nlSize];

    float *bestPosFeats = new float[posSize];
    float *bestNegFeats = new float[negSize];
    float rates[32];

    init_rates(rates, depth);

    memset(pairs, 0, sizeof(NodePair) * nlSize);

    for(int i = 0; i < nlSize; i++){
        pairs[i].posIdxs = new int[posSize];
        pairs[i].pws = new double[posSize];

        pairs[i].negIdxs = new int[negSize];
        pairs[i].nws = new double[negSize];
    }

    pairs[0].posSize = posSize;
    pairs[0].negSize = negSize;

    for(int i = 0; i < posSize; i++){
        pairs[0].posIdxs[i] = i;
        pairs[0].pws[i] = pws[i];
    }

    for(int i = 0; i < negSize; i++){
        pairs[0].negIdxs[i] = i;
        pairs[0].nws[i] = nws[i];
    }

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

        generate_feature_templates(posSet->winw, posSet->winh, sfts, featNum * FEAT_TIMES);

        FeatTemp bestFt;
        float bestError = FLT_MAX;
        float bestThresh = 0;

        for(int k = 0; k < FEAT_TIMES; k++){
            extract_features(posSet, posIdxs, posSize, sfts + k * featNum, featNum, posFeats);
            extract_features(negSet, negIdxs, negSize, sfts + k * featNum, featNum, negFeats);

#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
            for(int j = 0; j < featNum; j++){
                double error = FLT_MAX;
                float thresh;

                binary_classify_error(posFeats + j * posSize, tpws, posSize, negFeats + j * negSize, tnws, negSize, rate, thresh, error);

#pragma omp critical
                {
                    if(error < bestError){
                        bestError = error;
                        bestThresh = thresh;

                        memcpy(&bestFt, sfts + j + k * featNum, sizeof(FeatTemp));
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

        split(bestPosFeats, bestNegFeats, bestThresh, pairs + i, pairs + idL, pairs + idR);
    }

    double error = 0.0f;

    for(int i = nodeSize; i < nlSize; i++){
        Node *leaf = root + i;
        NodePair *pair = pairs + i;

        assert(pair->posSize > 0 && pair->negSize > 0);
        double sumpw = 0.0, sumnw = 0.0;

        for(int j = 0; j < pair->posSize; j++)
            sumpw += pair->pws[j];

        for(int j = 0; j < pair->negSize; j++)
            sumnw += pair->nws[j];

        leaf->score = log((sumpw + 0.000001) / (sumnw + 0.000001)) * 0.5f;

        if(leaf->score > 0)
            error += sumnw;
        else
            error += sumpw;

        leaf->pw = sumpw;
        leaf->nw = sumnw;
        leaf->posSize = pair->posSize;
        leaf->negSize = pair->negSize;
    }

    for(int i = 0; i < nlSize; i++){
        delete [] pairs[i].negIdxs;
        delete [] pairs[i].posIdxs;

        delete [] pairs[i].nws;
        delete [] pairs[i].pws;
    }

    delete [] pairs;
    delete [] sfts;
    delete [] bestPosFeats;
    delete [] bestNegFeats;

    return error;
}


float predict(Tree *root, int depth, uint32_t *iImg, int istride){
    for(int i = 0; i < depth - 1; i++){
        float feat;
        FeatTemp *ptrFeat = &root->ft;
        GET_VALUE(ptrFeat, iImg, istride, feat);
        root = root->lchild + (feat > root->thresh);
    }

    assert(root != NULL);

    return root->score;
}


void print_tree(Tree* root, int depth, FILE *fout){
    int nlSize = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);
    int nodeSize = nlSize - leafSize;

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
        }
    }
}


