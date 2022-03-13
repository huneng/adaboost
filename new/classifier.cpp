#include "classifier.h"


static void expand_space(StrongClassifier *sc){
    int len = 20;
    if(sc->capacity == 0){
        sc->capacity = len;
        sc->treeSize = 0;

        sc->trees = new Tree*[sc->capacity];

        memset(sc->trees, 0, sizeof(Tree *) * sc->capacity);
    }
    else if(sc->treeSize == sc->capacity){
        int capacity = sc->capacity + len;
        Tree **trees = new Tree*[capacity];

        memset(trees, 0, sizeof(Tree*) * capacity);
        memcpy(trees, sc->trees, sizeof(Tree*) * sc->treeSize);

        delete [] sc->trees;
        sc->trees = trees;

        sc->capacity = capacity;
    }
}


static float predict(StrongClassifier *sc, uint32_t *intImg, int istride){
    Tree **ptrTrees = sc->trees;

    int i = 0;
    int depth = sc->depth;
    int treeSize = sc->treeSize;

    float score = 0;

    for(; i <= treeSize - 8; i += 8, ptrTrees += 8){
        score += predict(ptrTrees[0], depth, intImg, istride);
        score += predict(ptrTrees[1], depth, intImg, istride);
        score += predict(ptrTrees[2], depth, intImg, istride);
        score += predict(ptrTrees[3], depth, intImg, istride);

        score += predict(ptrTrees[4], depth, intImg, istride);
        score += predict(ptrTrees[5], depth, intImg, istride);
        score += predict(ptrTrees[6], depth, intImg, istride);
        score += predict(ptrTrees[7], depth, intImg, istride);
    }

    for(; i < treeSize; i++, ptrTrees ++)
        score += predict(ptrTrees[0], depth, intImg, istride);

    return score;
}


static float get_precision(StrongClassifier *sc, SampleSet *posSet, double *pws, SampleSet *negSet, double *nws, float recall, float &thresh){
    int posSize = posSet->ssize;
    int negSize = negSet->ssize;

    int fsize = sc->treeSize;

    float *posScores = new float[posSize];
    float *negScores = new float[negSize];

    memset(posScores, 0, sizeof(float) * posSize);
    memset(negScores, 0, sizeof(float) * negSize);

    for(int i = 0; i < posSize; i++){
        Sample *sample = posSet->samples[i];
        posScores[i] = predict(sc, sample->iImg, sample->istride);

        pws[i] = exp(-posScores[i]);
    }

    for(int i = 0; i < negSize; i++){
        Sample *sample = negSet->samples[i];
        negScores[i] = predict(sc, sample->iImg, sample->istride);

        nws[i] = exp(negScores[i]);
    }

    sort_arr_float(posScores, posSize);
    sort_arr_float(negScores, negSize);

    recall = 1.0f - recall;

    thresh = posScores[int(recall * posSize)] - 0.000001f;


    int pID = 0;
    for( ; pID < posSize; pID++){
        if(posScores[pID] > thresh)
            break;
    }

    int nID = 0;
    for(; nID < negSize; nID++){
        if(negScores[nID] > thresh)
            break;
    }

    float TP = 1.0f - float(pID) / posSize;
    float FP = 1.0f - float(nID) / negSize;

    printf("%f %f\n", TP, FP);

    delete [] posScores;
    delete [] negScores;

    return (TP) / (TP + FP);
}


int train(StrongClassifier *sc, SampleSet *posSet, SampleSet *negSet, TrainParams *params){
    int WINW = posSet->winw;
    int WINH = posSet->winh;

    int posSize = posSet->ssize;
    int negSize = negSet->ssize;

    float *posFeats, *negFeats;
    double *pws, *nws;

    float rate = 0.0f;

    printf("GENERATE FEATURE TEMPLATE\n");

    sc->depth = params->depth;

    int nlSize = (1 << sc->depth) - 1;

    posFeats = new float[posSize * params->featNum]; assert(posFeats != NULL);
    negFeats = new float[negSize * params->featNum]; assert(negFeats != NULL);

    pws = new double[posSize]; assert(pws != NULL);
    nws = new double[negSize]; assert(nws != NULL);

    cv::RNG rng(cv::getTickCount());

    for(int i = 0; i < posSize; i++)
        pws[i] = 1.0;

    for(int i = 0; i < negSize; i++)
        nws[i] = 1.0;

    FILE *flog = fopen("classifier.txt", "w");
    int cId = 0;

    while(rate < params->precision && sc->treeSize < 200){
        printf("binary tree: %d\n", cId); fflush(stdout);
        fprintf(flog, "classifier: %d\n", cId ++);

        expand_space(sc);

        update_weights(pws, posSize);
        update_weights(nws, negSize);

        if(sc->trees[sc->treeSize] == NULL){
            sc->trees[sc->treeSize] = new Node[nlSize];
            memset(sc->trees[sc->treeSize], 0, sizeof(Node) * nlSize);
        }

        train(sc->trees[sc->treeSize], sc->depth,
                posSet, posFeats, pws,
                negSet, negFeats, nws, params->featNum);

        print_tree(sc->trees[sc->treeSize], sc->depth, flog); fflush(flog);

        sc->treeSize ++;

        rate = get_precision(sc, posSet, pws, negSet, nws, params->recall, sc->thresh);

        fprintf(flog, "sc thresh: %f precision: %f\n\n", sc->thresh, rate); fflush(flog);
        printf("sc thresh: %f precision: %f\n\n", sc->thresh, rate);
    }

    printf("\n");
    fclose(flog);

    delete [] posFeats, delete [] negFeats;
    delete [] pws, delete [] nws;

    return 0;
}



int predict(StrongClassifier *sc, uint32_t *intImg, int istride, float &score){
    Tree **ptrTrees = sc->trees;

    int i = 0;
    int depth = sc->depth;
    int treeSize = sc->treeSize;

    score = 0;

    for(i = 0; i < treeSize; i++){
        score += predict(ptrTrees[i], depth, intImg, istride);
    }
    if(score <= sc->thresh)
        return 0;

    return 1;
}


int predict(StrongClassifier *sc, int scSize, uint32_t *intImg, int istride, float &score){
    score = 0;
    for(int i = 0; i < scSize; i++){
        StrongClassifier *ptrSC = sc + i;

        Tree **ptrTrees = ptrSC->trees;
        int depth = ptrSC->depth;
        int treeSize = ptrSC->treeSize;

        float sscore = 0;
        int j;

        for(j = 0; j < treeSize; j++){
            sscore += predict(ptrTrees[j], depth, intImg, istride);
        }
        if(sscore <= ptrSC->thresh)
            return 0;

        score = sscore;
    }

    return 1;
}


void get_feature_template(StrongClassifier *scs, int scSize, FeatTemp **fts, int &top){
    top = 0;

    if(fts == NULL){
        for(int s = 0; s < scSize; s++){
            StrongClassifier *sc = scs + s;

            int nlSize   = (1 << sc->depth) - 1;
            int leafSize = 1 << (sc->depth - 1);

            int nodeSize = nlSize - leafSize;

            top += nodeSize * sc->treeSize;
        }
    }
    else {
        for(int s = 0; s < scSize; s++){
            StrongClassifier *sc = scs + s;

            int nlSize   = (1 << sc->depth) - 1;
            int leafSize = 1 << (sc->depth - 1);

            int nodeSize = nlSize - leafSize;

            for(int i = 0; i < sc->treeSize; i++){
                Tree *tree = sc->trees[i];

                for(int j = 0; j < nodeSize; j++){
                    fts[top] = &tree[j].ft;
                    top++;
                }
            }
        }
    }
}


int save(StrongClassifier *sc, FILE *fout){
    if(fout == NULL || sc == NULL)
        return 1;

    int ret;

    ret = fwrite(&sc->treeSize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&sc->depth, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&sc->thresh, sizeof(float), 1, fout); assert(ret == 1);

    for(int i = 0; i < sc->treeSize; i++)
        save(sc->trees[i], sc->depth, fout);

    return 0;
}


int load(StrongClassifier *sc, FILE *fin){
    if(fin == NULL || sc == NULL){
        return 1;
    }

    int ret;
    int nlSize;

    ret = fread(&sc->treeSize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&sc->depth, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&sc->thresh, sizeof(float), 1, fin); assert(ret == 1);

    sc->trees = new Tree*[sc->treeSize];

    nlSize = (1 << sc->depth) - 1;

    sc->capacity  = sc->treeSize;

    for(int i = 0; i < sc->treeSize; i++){
        sc->trees[i] = new Node[nlSize];
        memset(sc->trees[i], 0, sizeof(Node) * nlSize);
        load(sc->trees[i], sc->depth, fin);
    }

    return 0;
}


void release(StrongClassifier **sc){
    if(sc == NULL)
        return;

    release_data(*sc);

    delete *sc;
    *sc = NULL;
}


void release_data(StrongClassifier *sc){
    if(sc != NULL){
        if(sc->trees != NULL){
            for(int i = 0; i < sc->treeSize; i++){
                if(sc->trees[i] != NULL)
                    delete [] sc->trees[i];
                sc->trees[i] = NULL;
            }

            delete [] sc->trees;
        }

        sc->trees = NULL;
    }

    sc->capacity = 0;
    sc->treeSize = 0;
}
