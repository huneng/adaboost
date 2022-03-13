OBJ = adaboost.o classifier.o tree.o sample.o tool.o
OUTDIR=bin

FLAGS = -O3 -fopenmp $(shell pkg-config --cflags opencv)
LIBS  = $(shell pkg-config  --libs opencv) 
DST = $(OUTDIR)/train $(OUTDIR)/idetect $(OUTDIR)/vdetect $(OUTDIR)/rate

all: outdir $(DST)

outdir:
	mkdir -p $(OUTDIR)

$(OUTDIR)/train:main.cpp $(OBJ)
	g++ -DMAIN_TRAIN $^ -o $@ $(FLAGS) $(LIBS)

$(OUTDIR)/vdetect:main.cpp $(OBJ)
	g++ -DMAIN_DETECT_VIDEOS $^ -o $@ $(FLAGS) $(LIBS)

$(OUTDIR)/idetect:main.cpp $(OBJ)
	g++ -DMAIN_DETECT_IMAGES $^ -o $@ $(FLAGS) $(LIBS)

$(OUTDIR)/rate:main.cpp $(OBJ)
	g++ -DMAIN_DETECT_IMAGES_FDDB $^ -o $@ $(FLAGS) $(LIBS)

$(OBJ): %.o: %.cpp
	g++ $(FLAGS) -c $^

clean:
	@rm $(OBJ) -f
