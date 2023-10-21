#include <fstream>
#include <ctime>
#include <random>
#include <vector>
#include <string>

namespace RandomGeneration {
	void init() {
		srand(unsigned(clock()));
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

class SerialMatrixVectorMult {
public:
	SerialMatrixVectorMult(int size)
		: mMatrix(size * size), mVector(size), mResult(size)
	{

	}

	void randomDataInitialization() {
		auto size = mVector.size();
		for (size_t i = 0; i < size; i++) {
			mVector[i] = RandomGeneration::randDoubleValue<-1000, 1000>();
			for (size_t j = 0; j < size; j++) {
				mMatrix[i * size + j] = RandomGeneration::randDoubleValue<-1000, 1000>();
			}
		}
	}

	void resultCalculation() {
		auto size = mResult.size();
		for (size_t i = 0; i < size; i++) {
			mResult[i] = 0;
			for (size_t j = 0; j < size; j++) {
				mResult[i] += mMatrix[i * size + j] * mVector[j];
			}
		}
	}

private:
	using DataStorage = std::vector<double>;
private:
	DataStorage mMatrix;
	DataStorage mVector;
	DataStorage mResult;
};

void experiment(std::ofstream& ostrm, int size) {
	SerialMatrixVectorMult mult(size);
	mult.randomDataInitialization();

	auto start = clock();
	mult.resultCalculation();
	auto finish = clock();
	auto duration = (finish - start) / double(CLOCKS_PER_SEC);

	ostrm << "Size = " << size << ", Time = " << duration << std::endl;
}

void main()
{
	std::string fileName = "SerialMV";
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
	for (int i = 1000; i <= 10000; i = i + 1000) {
		experiment(ostrm, i);
	}
}
