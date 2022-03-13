OBJ = adaboost.o classifier.o tree.o sample.o tool.o
OUTDIR=bin

FLAGS = -O3 -fopenmp -I/usr/local/include -I/usr/include
LIBS  =  -L/usr/lib -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui
DST = $(OUTDIR)/train $(OUTDIR)/idetect $(OUTDIR)/vdetect

all: outdir $(DST)

outdir:
	mkdir -p $(OUTDIR)

$(OUTDIR)/train:main.cpp $(OBJ)
	g++ -DMAIN_TRAIN $^ -o $@ $(FLAGS) $(LIBS)

$(OUTDIR)/vdetect:main.cpp $(OBJ)
	g++ -DMAIN_DETECT_VIDEOS $^ -o $@ $(FLAGS) $(LIBS)

$(OUTDIR)/idetect:main.cpp $(OBJ)
	g++ -DMAIN_DETECT_IMAGES $^ -o $@ $(FLAGS) $(LIBS)

$(OBJ): %.o: %.cpp
	g++ $(FLAGS) -c $^

clean:
	@rm $(OBJ)
