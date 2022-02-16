
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <unistd.h>
#include <cstdlib>

#include "algorithms/SynC.h"
#include "algorithms/GPU_SynC.cuh"

class Data
{
public:
    float *data = nullptr;
    int n = 0;
    int d = 0;

    Data(std::string file)
    {
        std::ifstream in(file);
        std::vector<std::vector<float>> fields;

        if (in)
        {
            std::string line;

            while (std::getline(in, line))
            {
                std::stringstream sep(line);
                std::string field;

                fields.push_back(std::vector<float>());

                while (getline(sep, field, ','))
                {
                    fields.back().push_back(stof(field));
                }
            }
        }
        else
        {
            printf("data not found!");
            return;
        }

        n = fields.size();
        if (n > 0)
        {
            d = fields[0].size();
            if (d > 0)
            {
                data = new float[n * d];

                int i = 0;
                for (auto row : fields)
                {
                    for (auto e : row)
                    {
                        data[i] = e;
                        i++;
                    }
                }
            }
        }
    }
};

int main(int argc, char **argv)
{




    char tmp[256];
    getcwd(tmp, 256);
    std::cout << "Current working directory: " << tmp << std::endl;

    int n = 10000;
    int d = 2;
    int cl = 5;
    int v = 0;

    if(argc>1){
        n = std::stoi(argv[1]);
    }
    if(argc>2){
        d = std::stoi(argv[2]);
    }
    if(argc>3){
        cl = std::stoi(argv[3]);
    }
    if(argc>4){
        v = std::stoi(argv[4]);
    }

    std::string filename = "data/n"+std::to_string(n)+"d"+std::to_string(d)+"cl"+std::to_string(cl)+".csv";

    std::cout << "filename: " << filename << std::endl;

    Data data(filename);
    // Data data("data/iris.data");

    if(data.data == nullptr) {
        return 0;
    }

    GPU_DynamicalClustering_DOUBLE_GRID(data.data, data.n, data.d, 0.02, 1. - 0.001, 1, v);
//    GPU_DynamicalClustering(data.data, data.n, data.d, 0.02, 1. - 0.001);

    return 0;
}
