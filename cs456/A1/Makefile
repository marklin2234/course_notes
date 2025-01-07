CXX = clang++
CFLAGS = -Wall -g -std=c++20

CLIENT_TARGET = client
SERVER_TARGET = server

.PHONY: all
all: $(CLIENT_TARGET) $(SERVER_TARGET)

$(CLIENT_TARGET): client.cpp
	$(CXX) $(CFLAGS) client.cpp -o $@

$(SERVER_TARGET): server.cpp
	$(CXX) $(CFLAGS) server.cpp -o $@

.PHONY: clean
clean:
	rm -rf client server
