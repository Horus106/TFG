# WFLAGS := \
#   -Wall \
#   -Wcast-align \
#   -Wconversion \
#   -Wdisabled-optimization \
#   -Wdouble-promotion \
#   -Wduplicated-branches \
#   -Wduplicated-cond \
#   -Weffc++ \
#   -Wextra \
#   -Wfloat-equal \
#   -Wformat=2 \
#   -Winline \
#   -Wlogical-op \
#   -Wmisleading-indentation \
#   -Wnoexcept \
#   -Wnon-virtual-dtor \
#   -Wnull-dereference \
#   -Wold-style-cast \
#   -Woverloaded-virtual \
#   -Wpedantic \
#   -Wshadow \
#   -Wsign-conversion \
#   -Wsign-promo \
#   -Wunused \
#   -Wuseless-cast \
#   -Wzero-as-null-pointer-constant
SRC = src/
OBJ = obj/
INC = include/
OPENCV4 = /usr/include/opencv4
LOPENCV = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
COMPILER = g++ -O3 $(LOPENCV) $(WFLAGS)
FLAGS = -I$(INC) -I$(OPENCV4)
exe = bin/panorama_extended
units_cpp := $(wildcard $(addprefix $(SRC), *.cpp))
units_o := $(addprefix $(OBJ), $(addsuffix .o, $(notdir $(basename $(units_cpp)))))
headers := $(wildcard $(addprefix $(INC), *.h*))

x: $(exe)
	@echo Compilando $(exe)

mkdir:
	mkdir bin/ obj/

init: mkdir $(exe)
	@echo Compilando $(exe)

$(exe): $(units_o) makefile
	$(COMPILER) -o $(exe) $(units_o)

$(OBJ)%.o : $(SRC)%.cpp
	$(COMPILER) -o $@ -c $(FLAGS) $<

$(units_cpp) : $(headers)
	touch $(units_cpp)

doc:
	doxygen Doxyfile

clean:
	rm -rf obj/*.o bin/* doc/*
