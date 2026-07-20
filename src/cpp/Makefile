
CXX := g++ -g
PACKAGES := -Icpp_include -I/usr/local/include 
CXXFLAGS := -std=c++17 -Wall -Wextra -I. $(PACKAGES)

LDFLAGS := -L/usr/local/lib -lsentencepiece -lpqxx -lpq -lcurl -lyaml-cpp -lPocoFoundation

# Source files
INDEXER_SRC := indexer/indexer.cpp db_handler/db_handler.cpp 
DB_HANDLER_SRC := db_handler/db_handler.cpp
TEST_HANDLER_SRC := indexer/test_file.cpp

# Object files
INDEXER_OBJ := $(INDEXER_SRC:.cpp=.o)
DB_HANDLER_OBJ := $(DB_HANDLER_SRC:.cpp=.o)
TEST_FILE_OBJ := $(TEST_HANDLER_SRC:.cpp=.o)

# Create bin dir before linking
#MKDIR_BIN := mkdir -p bin

all: indexer db_handler

# Linking rules
indexer: $(INDEXER_OBJ)
	mkdir -p bin
	@echo "Linking $@ ..."
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o bin/$@

db_handler: $(DB_HANDLER_OBJ)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o bin/$@

test_file: $(TEST_FILE_OBJ)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o bin/$@

# Compile source files to objects
%.o: %.cpp
	@echo "Compiling $< ..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f bin/indexer bin/db_handler $(INDEXER_OBJ) $(DB_HANDLER_OBJ)
