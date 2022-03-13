#include "adaboost.h"

#if defined(WIN32)
#include <time.h>
#else
#include <sys/time.h>
#endif

int main_train(int argc, char **argv){
    if(argc < 4){
        printf("Usage: %s [pos list] [neg list] [out model]\n", argv[0]);
        return 1;
    }

    ObjectDetector objDetector;

    memset(&objDetector, 0, sizeof(ObjectDetector));

    int ret = train(&objDetector, argv[1], argv[2]);
    if(ret != 0){
        printf("TRAIN ERROR\n");
        return 2;
    }

    save(&objDetector, argv[3]);

    release_data(&objDetector);

    return 0;
}


int main_detect_images_fddb(int argc, char **argv){
    if(argc < 5){
        printf("Usage: %s [model] [image dir] [image list] [outdir]\n", argv[0]);
        return 1;
    }

    ObjectDetector objDetector;
    std::vector<std::string> imgList;
    int size;
    int ret;

    char filePath[256], rootDir[128], fileName[128], ext[30];

    FILE *fout;

    printf("load model\n");
    ret = load(&objDetector, argv[1]);

    if(ret != 0) return 1;

    init_detect_factor(&objDetector, 0.05, 1.0, 0.1, 30);

    split_file_path(argv[3], rootDir, fileName, ext);

    read_list(argv[3], imgList);

    sprintf(filePath, "%s/%s.%s", argv[4], fileName, ext);
    fout = fopen(filePath, "w"); assert(fout != NULL);

    size = imgList.size();

    printf("detect\n");

    for(int i = 0; i < size; i++){
        cv::Mat img, src, gray;
        HRect *rects = NULL;
        float *scores = NULL;
        int num;

        sprintf(filePath, "%s/%s.jpg", argv[2], imgList[i].c_str());

        img = cv::imread(filePath, 1);

        if(img.empty()){
            printf("Can't open image %s\n", imgList[i].c_str());
            continue;
        }

        src.create(img.rows * 1.4, img.cols * 1.4, img.type());
        src.setTo(0);

        src(cv::Rect(0.2 * img.cols, 0.2 * img.rows, img.cols, img.rows)) += img;

        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

        num = detect(&objDetector, gray.data, gray.cols, gray.rows, gray.step, &rects, &scores);

        fprintf(fout, "%s\n%d\n", imgList[i].c_str(), num);

        for(int j = 0; j < num; j++){
            HRect rect = rects[j];
            fprintf(fout, "%d %d %d %d %f\n", rect.x, rect.y, rect.width, rect.height, scores[j]);
        }

        if(num > 0){
            delete [] rects;
            delete [] scores;
        }

        rects = NULL;
    }

    fclose(fout);

    release_data(&objDetector);

    return 0;
}


int main_detect_images(int argc, char **argv){
    if(argc < 4){
        printf("Usage: %s [model] [image list] [outdir]\n", argv[1]);
        return 1;
    }

    ObjectDetector objDetector;
    std::vector<std::string> imgList;
    int size;

    printf("load model\n");
    int ret = load(&objDetector, argv[1]);

    if(ret != 0) return 1;

    init_detect_factor(&objDetector, 0.1, 0.9, 0.1, 20);
    read_list(argv[2], imgList);

    size = imgList.size();

    printf("detect\n");

    int finished = 0;
#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < size; i++){
        cv::Mat img = cv::imread(imgList[i], 1);
        cv::Mat gray;
        HRect *rects = NULL;
        float *scores = NULL;
        int num;

        char rootDir[256], fileName[256], ext[30], filePath[256];
        if(img.empty()){
            printf("Can't open image %s\n", imgList[i].c_str());
            continue;
        }

        cv::resize(img, img, cv::Size(720, 720 * img.rows / img.cols));
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        num = detect(&objDetector, gray.data, gray.cols, gray.rows, gray.step, &rects, &scores);

        split_file_path(imgList[i].c_str(), rootDir, fileName, ext);

#if 1
        for(int j = 0; j < num; j++){
            int x = HU_MAX(0, rects[j].x);
            int y = HU_MAX(0, rects[j].y);
            int w = HU_MIN(img.cols - x, rects[j].width);
            int h = HU_MIN(img.rows - y, rects[j].height);

            cv::Mat patch(img, cv::Rect(x, y, w, h));
            cv::resize(patch, patch, cv::Size(100, 100));

            sprintf(filePath, "%s/%s_%d.jpg", argv[3], fileName, j);

            cv::imwrite(filePath, patch);
        }
#pragma omp critical
        {
            finished ++;
            printf("%d\r", finished); fflush(stdout);
        }
#else
        sprintf(filePath, "%s/%s.png", argv[3], fileName);

        for(int j = 0; j < num; j++){
            cv::rectangle(img, cv::Rect(rects[j].x, rects[j].y, rects[j].width, rects[j].height), cv::Scalar(0, 255, 0), 2);
        }



        cv::imwrite(filePath, img);
#endif
        if(num > 0){
            delete [] rects;
            delete [] scores;
        }

        rects = NULL;
    }

    release_data(&objDetector);

    return 0;
}


int main_detect_videos(int argc, char **argv){
    if(argc < 3){
        printf("Usage: %s [model] [video]\n", argv[0]);
        return 1;
    }

    ObjectDetector objDetector;
    cv::VideoCapture cap;
    int totalFrame;

    printf("load model\n");
    int ret = load(&objDetector, argv[1]);

    if(ret != 0) return 1;

    init_detect_factor(&objDetector, 0.2, 1.0, 0.1, 12);

    cap.open(argv[2]);
    if(!cap.isOpened()){
        printf("Can't open video %s\n", argv[2]);
        return 2;
    }

    totalFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);

    printf("detect\n");

#if defined(WIN32)
    clock_t st, et;
#else
    struct timeval  stv, etv;
    struct timezone tz;
#endif

    for(int fIdx = 0; fIdx < totalFrame; fIdx++){
        cv::Mat frame;

        cap >> frame;
        if(frame.empty()) continue;

        cv::Mat gray;
        HRect *rects = NULL;
        float *scores = NULL;
        int num;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

#if defined(WIN32)
        st = clock();
#else
        gettimeofday(&stv, &tz);
#endif
        num = detect(&objDetector, gray.data, gray.cols, gray.rows, gray.step, &rects, &scores);
#if defined(WIN32)
        et = clock();
        printf("%.2f ms\n", (et - st) * 1000.0f / CLOCKS_PER_SEC);
#else
        gettimeofday(&etv, &tz);
        printf("%.2f ms\n", (etv.tv_sec - stv.tv_sec) * 1000.0f + (etv.tv_usec - stv.tv_usec) * 0.001);
#endif

        for(int j = 0; j < num; j++)
            cv::rectangle(frame, cv::Rect(rects[j].x, rects[j].y, rects[j].width, rects[j].height), cv::Scalar(0, 255, 0), 2);

        cv::imshow("video", frame);
        cv::waitKey(1);

        if(num > 0){
            delete [] rects;
            delete [] scores;
        }
        rects = NULL;
    }

    release_data(&objDetector);

    return 0;
}


int main(int argc, char **argv){
#if defined(MAIN_TRAIN)
    main_train(argc, argv);

#elif defined(MAIN_DETECT_VIDEOS)
    main_detect_videos(argc, argv);

#elif defined(MAIN_DETECT_IMAGES)
    main_detect_images(argc, argv);

#elif defined(MAIN_DETECT_IMAGES_FDDB)
    main_detect_images_fddb(argc, argv);

#endif

    return 0;
}
