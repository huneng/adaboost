#include "adaboost.h"
#include <sys/time.h>


int main_train(int argc, char **argv){
    if(argc < 4){
        printf("Usage: %s [pos list] [neg list] [out model]\n", argv[0]);
        return 1;
    }

    Cascade objDetector;

    int ret = train(&objDetector, argv[1], argv[2]);
    if(ret != 0){
        printf("TRAIN ERROR\n");
        return 2;
    }

    save(&objDetector, argv[3]);

    release_data(&objDetector);

    return 0;
}


int main_detect_images(int argc, char **argv){
    if(argc < 4){
        printf("Usage: %s [model] [image list] [outdir]\n", argv[1]);
        return 1;
    }

    Cascade objDetector;
    std::vector<std::string> imgList;
    int size;

    printf("load model\n");
    int ret = load(&objDetector, argv[1]);

    if(ret != 0) return 1;

    init_detect_factor(&objDetector, 0.1, 0.9, 0.1, 20);
    read_file_list(argv[2], imgList);

    size = imgList.size();

    printf("detect\n");

//#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < size; i++){
        cv::Mat img = cv::imread(imgList[i], 1);
        cv::Mat gray;
        HRect *rects = NULL;
        int num;

        struct timezone tz;
        struct timeval stv, etv;

        char rootDir[256], fileName[256], ext[30], filePath[256];
        if(img.empty()){
            printf("Can't open image %s\n", imgList[i].c_str());
            continue;
        }

        cv::resize(img, img, cv::Size(720, 720 * img.rows / img.cols));
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        gettimeofday(&stv, &tz);
        num = detect(&objDetector, gray.data, gray.cols, gray.rows, gray.step, &rects);
        gettimeofday(&etv, &tz);

        printf("%.2f ms\n", (etv.tv_usec - stv.tv_usec) / 1000.0f);

        for(int j = 0; j < num; j++){
            cv::rectangle(img, cv::Rect(rects[j].x, rects[j].y, rects[j].width, rects[j].height), cv::Scalar(0, 255, 0), 2);
        }

        analysis_file_path(imgList[i].c_str(), rootDir, fileName, ext);

        sprintf(filePath, "%s/%s.png", argv[3], fileName);

#if 0
        if(num > 0){
            cv::imwrite(filePath, img);
            delete [] rects;
        }

#else
        cv::imwrite(filePath, img);
        if(num > 0){
            delete [] rects;
        }

#endif
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

    Cascade objDetector;
    cv::VideoCapture cap;
    int totalFrame;

    printf("load model\n");
    int ret = load(&objDetector, argv[1]);

    if(ret != 0) return 1;

    init_detect_factor(&objDetector, 0.1, 0.9, 0.1, 20);

    cap.open(argv[2]);
    if(!cap.isOpened()){
        printf("Can't open video %s\n", argv[2]);
        return 2;
    }

    totalFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);

    printf("detect\n");

    struct timeval  stv, etv;
    struct timezone tz;

    for(int fIdx = 0; fIdx < totalFrame; fIdx++){
        cv::Mat frame;

        cap >> frame;
        if(frame.empty()) continue;

        cv::Mat gray;
        HRect *rects = NULL;
        int num;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        gettimeofday(&stv, &tz);
        num = detect(&objDetector, gray.data, gray.cols, gray.rows, gray.step, &rects);
        gettimeofday(&etv, &tz);

        printf("%.2f ms\n", (etv.tv_usec - stv.tv_usec) * 0.001f);

        for(int j = 0; j < num; j++)
            cv::rectangle(frame, cv::Rect(rects[j].x, rects[j].y, rects[j].width, rects[j].height), cv::Scalar(0, 255, 0), 2);

        cv::imshow("video", frame);
        cv::waitKey(1);

        if(num > 0)
            delete [] rects;
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

#endif

    return 0;
}
