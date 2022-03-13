#include "tool.h"


int read_file_list(const char *filePath, std::vector<std::string> &fileList)
{
    char line[512];
    FILE *fin = fopen(filePath, "r");

    if(fin == NULL){
        printf("Can't open file: %s\n", filePath);
        return -1;
    }

    while(fscanf(fin, "%s\n", line) != EOF){
        fileList.push_back(line);
    }

    fclose(fin);

    return 0;
}


void analysis_file_path(const char* filePath, char *rootDir, char *fileName, char *ext)
{
    int len = strlen(filePath);
    int idx = len - 1, idx2 = 0;

    while(idx >= 0){
        if(filePath[idx] == '.')
            break;
        idx--;
    }

    if(idx >= 0){
        strcpy(ext, filePath + idx + 1);
        ext[len - idx] = '\0';
    }
    else {
        ext[0] = '\0';
        idx = len - 1;
    }

    idx2 = idx;
    while(idx2 >= 0){
#ifdef WIN32
        if(filePath[idx2] == '\\')
#else
        if(filePath[idx2] == '/')
#endif
            break;

        idx2 --;
    }

    if(idx2 > 0){
        strncpy(rootDir, filePath, idx2);
        rootDir[idx2] = '\0';
    }
    else{
        rootDir[0] = '.';
        rootDir[1] = '\0';
    }

    strncpy(fileName, filePath + idx2 + 1, idx - idx2 - 1);
    fileName[idx - idx2 - 1] = '\0';
}


void integral_image(uint8_t *img, int width, int height, int stride, uint32_t *intImg, int istride){
    int id0 = 0, id1 = 0;

    for(int y = 0; y < height; y++){
        intImg[id0] = img[id1];
        for(int x = 1; x < width; x++){
            intImg[id0 + x] = img[id1 + x] + intImg[id0 + x - 1];
        }

        id0 += istride;
        id1 += stride;
    }

    id0 = 0, id1 = istride;

    for(int y = 1; y < height; y++){
        for(int x = 0; x < width; x++){
            intImg[id1 + x] += intImg[id0 + x];
        }

        id0 += istride;
        id1 += istride;
    }
}


void update_weights(double *weights, int size){
    double sum = 0.0;

    for(int i = 0; i < size; i++)
        sum += weights[i];

    sum = 1.0 / sum;
    for(int i = 0; i < size; i++){
        weights[i] *= sum;
    }

}


#define FIX_INTER_POINT 14

void resizer_bilinear_gray(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts){
    uint16_t *table = NULL;

    uint16_t FIX_0_5 = 1 << (FIX_INTER_POINT - 1);
    float scalex, scaley;

    scalex = srcw / float(dstw);
    scaley = srch / float(dsth);

    table = new uint16_t[dstw * 3];

    for(int i = 0; i < dstw; i++){
        float x = i * scalex;

        if(x < 0) x = 0;
        if(x > srcw - 1) x = srcw - 1;

        int x0 = int(x);

        table[i * 3] = x0;
        table[i * 3 + 2] = (x - x0) * (1 << FIX_INTER_POINT);
        table[i * 3 + 1] = (1 << FIX_INTER_POINT) - table[i * 3 + 2];
    }

    int sId = 0, dId = 0;

    for(int y = 0; y < dsth; y++){
        int x;
        float yc;

        uint16_t wy0, wy1;
        uint16_t y0, y1;
        uint16_t *ptrTab = table;
        int buffer[8];
        yc = y * scaley;
        yc = yc > 0 ? yc : 0;
        yc = yc < srch - 1 ? yc : srch - 1;

        y0 = uint16_t(yc);
        y1 = y0 + 1;

        wy1 = uint16_t((yc - y0) * (1 << FIX_INTER_POINT));
        wy0 = (1 << FIX_INTER_POINT) - wy1;

        sId = y0 * srcs;

        uint8_t *ptrDst = dst + dId;

        for(x = 0; x <= dstw - 4; x += 4){
            uint16_t x0, x1, wx0, wx1;
            uint8_t *ptrSrc0, *ptrSrc1;

            //1
            x0 = ptrTab[0], x1 = x0 + 1;
            wx0 = ptrTab[1], wx1 = ptrTab[2];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[0] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[1] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //2
            x0 = ptrTab[3], x1 = x0 + 1;

            wx0 = ptrTab[4], wx1 = ptrTab[5];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[2] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[3] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //3
            x0 = ptrTab[6], x1 = x0 + 1;

            wx0 = ptrTab[7], wx1 = ptrTab[8];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[4] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[5] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //4
            x0 = ptrTab[9], x1 = x0 + 1;
            wx0 = ptrTab[10], wx1 = ptrTab[11];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[6] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[7] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrDst[0] = (wy0 * (buffer[0] - buffer[1]) + (buffer[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[1] = (wy0 * (buffer[2] - buffer[3]) + (buffer[3] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[2] = (wy0 * (buffer[4] - buffer[5]) + (buffer[5] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[3] = (wy0 * (buffer[6] - buffer[7]) + (buffer[7] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrDst += 4;
            ptrTab += 12;
        }

        for(; x < dstw; x++){
            uint16_t x0, x1, wx0, wx1, valuex0, valuex1;

            uint8_t *ptrSrc0, *ptrSrc1;
            x0 = ptrTab[0], x1 = x0 + 1;

            wx0 = ptrTab[1], wx1 = ptrTab[2];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            valuex0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            valuex1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            dst[y * dsts + x] = (wy0 * (valuex0 - valuex1) + (valuex1 << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrTab += 3;
        }

        dId += dsts;
    }


    delete [] table;
}


#define BILINEAR_INTER(res, imgData, width, height, stride, x, y) \
{ \
    int X0, Y0;                                         \
    float wx, wy, vx0, vx1;                             \
    uint8_t *ptrData;                                   \
                                                        \
    X0 = int(x);                                        \
    Y0 = int(y);                                        \
    wx = x - X0;                                        \
    wy = y - Y0;                                        \
                                                        \
    ptrData = imgData + Y0 * stride + X0;               \
    vx0 = ptrData[0] * (1.0f - wx) + ptrData[1] * wx;   \
                                                        \
    ptrData += stride;                                  \
    vx1 = ptrData[0] * (1.0f - wx) + ptrData[1] * wx;   \
                                                        \
    res = vx0 * (1 - wy) + vx1 * wy;                    \
}


void affine_image(uint8_t *src, int srcw, int srch, int srcs, float angle, float scale, cv::Point2f &center,
        uint8_t *dst, int dstw, int dsth, int dsts){
    float sina = sin(-angle) / scale;
    float cosa = cos(-angle) / scale;

    int id = 0;

    float *xtable = new float[dstw << 1]; assert(xtable != NULL);
    float *ytable = new float[dsth << 1]; assert(ytable != NULL);

    float cx = (float)dstw / 2;
    float cy = (float)dsth / 2;

    for(int i = 0; i < dsth; i++){
        int idx = i << 1;

        float y0 = (i - cy);

        ytable[idx]     = y0 * sina;
        ytable[idx + 1] = y0 * cosa;
    }

    for(int i = 0; i < dstw; i++){
        int idx = i << 1;

        float x0 = (i - cx);

        xtable[idx]     = x0 * sina;
        xtable[idx + 1] = x0 * cosa;
    }

    id = 0;
    for(int y = 0; y < dsth; y++){
        int idx = y << 1;
        float ys = ytable[idx]     + center.x;
        float yc = ytable[idx + 1] + center.y;

        for(int x = 0; x < dstw; x++){
            idx = x << 1;

            float xs = xtable[idx];
            float xc = xtable[idx + 1];

            float x1 =  xc + ys;
            float y1 = -xs + yc;

            uint8_t value = 0;

            if(0 <= x1 && x1 < srcw && 0 <= y1 && y1 < srch)
                BILINEAR_INTER(value, src, srcw, srch, srcs, x1, y1);

            dst[id + x] = (uint8_t)value;
        }

        id += dsts;
    }

    delete [] xtable;
    delete [] ytable;
}


void transform_image(uint8_t *img, int width, int height, int stride, uint8_t *dImg){
    static cv::RNG rng(cv::getTickCount());

    float angle = rng.uniform(-HU_PI, HU_PI);
    float scale = rng.uniform(0.7, 1.3);

    assert(stride * height < 4096 * 4096);

    cv::Point2f center(width >> 1, height >> 1);

    memset(dImg, 0, sizeof(uint8_t) * height * stride);

    affine_image(img, width, height, stride, angle, scale, center, dImg, width, height, stride);

    memcpy(img, dImg, height * stride);

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img;

        float sum = 0;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++)
                sum += data[x];

            data += stride;
        }

        sum /= (width * height);
        sum ++;

        data = img;
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++)
                data[x] = ((data[x] - sum) / (data[x] + sum) + 1.0f) * 0.5 * 255;

            data += stride;
        }
    }

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                if(data[x] > 16 && data[x] < 239)
                    data[x] += rng.uniform(-8, 8);
            }

            data += stride;
        }
    }

    if(rng.uniform(0, 8) == 0)
        vertical_mirror(img, width, height, stride);

    if(rng.uniform(0, 8) == 0)
        horizontal_mirror(img, width, height, stride);

    if(rng.uniform(0, 8) == 0){
        cv::Mat sImg(height, width, CV_8UC1, img);
        cv::Mat blur;

        cv::GaussianBlur(sImg, blur, cv::Size(5, 5), 0, 0);

        for(int y = 0; y < height; y++)
            memcpy(img + y * stride, blur.data + y * blur.step, sizeof(uint8_t) * width);
    }


    if(rng.uniform(0, 8) == 0){
        cv::Mat sImg(height, width, CV_8UC1, img);
        cv::Mat blur;

        cv::equalizeHist(sImg, blur);

        for(int y = 0; y < height; y++)
            memcpy(img + y * stride, blur.data + y * blur.step, sizeof(uint8_t) * width);
    }
}


void transform_image(cv::Mat &img, int WINW){
    static cv::RNG rng(cv::getTickCount());

    int w = rng.uniform(3 * WINW, 10 * WINW);
    int h = w * img.rows / img.cols;

    assert(w > 0 && h > 0);
    float angle = rng.uniform(-180.0f, 180.0f);

    cv::Mat affineMat = cv::getRotationMatrix2D(cv::Point2f(img.cols >> 1, img.rows >> 1), angle, float(w) / img.cols);

    cv::warpAffine(img, img, affineMat, cv::Size(w, h));

    if(rng.uniform(0, 8) == 0){
        cv::equalizeHist(img, img);
    }

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img.data;

        float sum = 0;
        for(int y = 0; y < img.rows; y++){
            for(int x = 0; x < img.cols; x++)
                sum += data[x];

            data += img.step;
        }

        sum /= (img.cols * img.rows);
        sum += 1;

        data = img.data;
        for(int y = 0; y < img.rows; y++){
            for(int x = 0; x < img.cols; x++)
                data[x] = ((data[x] - sum) / (data[x] + sum) + 1.0f) * 0.5 * 255;

            data += img.step;
        }
    }

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img.data;

        for(int y = 0; y < img.rows; y++){
            for(int x = 0; x < img.cols; x++){
                if(data[x] > 16 && data[x] < 239)
                    data[x] += rng.uniform(-16, 16);
            }

            data += img.step;
        }
    }
}


void vertical_mirror(uint8_t *img, int width, int height, int stride)
{
    int cy = height / 2;

    for(int y = 0; y < cy; y++){
        uint8_t *ptr1 = img + y * stride;
        uint8_t *ptr2 = img + (height - y - 1) * stride;

        for(int x = 0; x < width; x++)
            HU_SWAP(ptr1[x], ptr2[x], uint8_t);
    }
}


void horizontal_mirror(uint8_t *img, int width, int height, int stride){
    int cx = width / 2;
    uint8_t *ptr = img;

    for(int y = 0; y < height; y++){
        for(int x = 0; x < cx; x++)
            HU_SWAP(ptr[x], ptr[width - x - 1], uint8_t);
        ptr += stride;
    }
}


#define LT(a, b) ((a) < (b))

IMPLEMENT_QSORT(sort_arr_float, float, LT);


void sleep(uint64_t milisecond){
    clock_t st = clock();
    clock_t et = st + milisecond * CLOCKS_PER_SEC / 1000;

    while(clock() < et){}
}
