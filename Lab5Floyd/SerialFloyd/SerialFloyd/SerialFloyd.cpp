#include <vector>
#include <fstream>
#include <ctime>
#include <random>
#include <algorithm>
#include <string>

namespace RandomGeneration {
    void init() {
        std::srand(unsigned(clock()));
    }

    template<int lowerBound, int upperBound>
    double mersenneDoubleValue() {
        static std::random_device randomDevice;
        static std::mt19937 generator(randomDevice());
        static std::uniform_real_distribution<> distribution(lowerBound, upperBound);
        return distribution(generator);
    }

    template<int lowerBound, int upperBound>
    int mersenneIntValue() {
        static std::random_device randomDevice;
        static std::mt19937 generator(randomDevice());
        static std::uniform_int_distribution<> distribution(lowerBound, upperBound);
        return distribution(generator);
    }

    template<int lowerBound, int upperBound>
    double randDoubleValue() {
        double f = (double)rand() / RAND_MAX;
        return lowerBound + f * (upperBound - lowerBound);
    }

    template<int lowerBound, int upperBound>
    int randIntValue() {
        double f = (double)rand() / RAND_MAX;
        return lowerBound + static_cast<int>(f * (upperBound - lowerBound));
    }
};

class SerialFloyd {
public:
    SerialFloyd(int size, double infinitiesPercent = 50.0)
        : mData(size * size)
        , mSize(size)
        , mInfinitiesPercent(infinitiesPercent)
    {
        randomDataInitialization();
    }

    void serialFloyd() {
        for (size_t k = 0; k < mSize; k++) {
            for (size_t i = 0; i < mSize; i++) {
                for (size_t j = 0; j < mSize; j++) {
                    if (mData[i * mSize + k] != -1 && mData[k * mSize + j] != -1) {
                        mData[i * mSize + j] = findMin(mData[i * mSize + j], mData[i * mSize + k] + mData[k * mSize + j]);
                    }
                }
            }
        }
    }

private:
    void randomDataInitialization() {
        using namespace RandomGeneration;
        for (int i = 0; i < mSize; i++) {
            for (int j = 0; j < mSize; j++) {
                if (i != j) {
                    mData[i * mSize + j] = randDoubleValue<0, 100>() < mInfinitiesPercent ? -1 : randIntValue<1, 1000>();
                }
                else {
                    mData[i * mSize + j] = 0;
                }
            }
        }
    }

    int findMin(int lhs, int rhs) {
        int result = lhs < rhs ? lhs : rhs;

        if (lhs < 0 && rhs >= 0) {
            result = rhs;
        }
        else if (rhs < 0 && lhs >= 0) {
            result = lhs;
        }
        if (lhs < 0 && rhs < 0) {
            result = -1;
        }

        return result;
    }
private:
    using DataStorage = std::vector<int>;
private:
    DataStorage mData;
    int mSize;
    double mInfinitiesPercent;
};

void experiment(std::ofstream& ostrm, int size) {
    SerialFloyd floyd(size);

    auto start = clock();
    floyd.serialFloyd();
    auto finish = clock();
    auto duration = (finish - start) / double(CLOCKS_PER_SEC);
    ostrm << "Size = " << size << ", Time = " << duration << std::endl;
}

void main()
{
    std::string fileName = "SerialFloyd";
#ifdef NDEBUG
    fileName += "Release";
#else
    fileName += "Debug";
#endif
    fileName += ".txt";

    std::ofstream ostrm(fileName, std::ios::out);

    RandomGeneration::init();
    experiment(ostrm, 10);
    experiment(ostrm, 100);
    for (int i = 500; i <= 1000; i += 100) {
        experiment(ostrm, i);
    }
}