#include "csv.hpp"
#include <iostream>

int main() {
    csv::CSVReader reader("sample.csv");
    for (auto& row : reader) {
        std::cout << row[0].get<>() << std::endl;
    }
    return 0;
}
