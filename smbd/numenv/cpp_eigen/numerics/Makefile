
BUILD := build/
SRC := src/smbd/

CC := g++
INC := -I src

All := $(BUILD)spatial_algebra.o $(BUILD)euler_parameters.o \
$(BUILD)helpers.o $(BUILD)utilities.o $(BUILD)geometries.o


all: $(All)

$(BUILD)spatial_algebra.o : $(SRC)spatial_algebra.cpp $(SRC)spatial_algebra.hpp
	$(CC) $(INC) -c -o $@ $<

$(BUILD)euler_parameters.o : $(SRC)euler_parameters.cpp $(SRC)euler_parameters.hpp
	$(CC) $(INC) -c -o $@ $<

$(BUILD)helpers.o : $(SRC)helpers.cpp $(SRC)helpers.hpp
	$(CC) $(INC) -c -o $@ $<

$(BUILD)utilities.o : $(SRC)utilities.cpp $(SRC)utilities.hpp
	$(CC) $(INC) -c -o $@ $<

$(BUILD)geometries.o : $(SRC)geometries.cpp $(SRC)geometries.hpp
	$(CC) $(INC) -c -o $@ $<


clear:
	rm $(BUILD)*.o
