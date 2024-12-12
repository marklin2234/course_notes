CXX = g++-14.2.0 -std=c++20 -fmodules-ts -Wall -g

OBJECTS = list.o list-impl.o tierlist.o tierlist-impl.o

TARGET = tierlist 

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) a3q3.cc $^ -o $@

list.o: list.cc
	$(CXX) -c $<

list-impl.o: list-impl.cc
	$(CXX) -c $<

tierlist.o: tierlist.cc
	$(CXX) -c $<

tierlist-impl.o: tierlist-impl.cc
	$(CXX) -c $<

clean:
	rm -rf *.o $(TARGET)

.PHONY: all clean

