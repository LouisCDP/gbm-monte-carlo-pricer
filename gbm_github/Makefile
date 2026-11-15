CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -pthread -Wall -Wextra

gbm_mc: src/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: gbm_mc
	./gbm_mc 1000000

clean:
	rm -f gbm_mc
