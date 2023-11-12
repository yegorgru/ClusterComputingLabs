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
	double randDoubleValue() {
		double f = (double)rand() / RAND_MAX;
		return lowerBound + f * (upperBound - lowerBound);
	}
};

class SerialGaussSeidel {
public:
	SerialGaussSeidel(int size, double eps = 1)
		: mData(size * size)
		, mSize(size)
		, mEps(eps)
	{
		randomDataInitialization();
	}

	int resultCalculation() {
		double dmax = 0;
		int iterations = 0;
		do {
			dmax = 0;
			for (int i = 1; i < mSize - 1; i++) {
				for (int j = 1; j < mSize - 1; j++) {
					auto temp = mData[mSize * i + j];
					mData[mSize * i + j] = 0.25 * (mData[mSize * i + j + 1] + 
												   mData[mSize * i + j - 1] +
												   mData[mSize * (i + 1) + j] +
												   mData[mSize * (i - 1) + j]
												  );
					auto dm = std::fabs(mData[mSize * i + j] - temp);
					dmax = std::max(dmax, dm);
				}
			}
			iterations++;
		} while (dmax > mEps);
		return iterations;
	}

private:
	void randomDataInitialization() {
		for (int i = 0; i < mSize; i++) {
			for (int j = 0; j < mSize; j++) {
				if (i == 0 || i == mSize - 1 || j == 0 || j == mSize - 1) {
					mData[i * mSize + j] = 100;
				}
				else {
					mData[i * mSize + j] = RandomGeneration::randDoubleValue<0, 28>();
				}
			}
		}
	}
private:
	using DataStorage = std::vector<int>;
private:
	DataStorage mData;
	int mSize;
	double mEps;
};

void experiment(std::ofstream& ostrm, int size) {
	SerialGaussSeidel sorter(size);

	auto start = clock();
	auto iters = sorter.resultCalculation();
	auto finish = clock();
	auto duration = (finish - start) / double(CLOCKS_PER_SEC);
	ostrm << "Size = " << size << ", Time = " << duration << " Number of iterations: " << iters << std::endl;
}

void main()
{
	std::string fileName = "SerialGaussSeidel";
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
	for (int i = 1000; i <= 10000; i += 1000) {
		experiment(ostrm, i);
	}
}