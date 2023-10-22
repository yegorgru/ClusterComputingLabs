#include <vector>
#include <fstream>
#include <ctime>
#include <random>
#include <iostream>

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

class SerialGauss {
public:
	SerialGauss(int size)
		: mMatrix(size * size), mVector(size), mResult(size)
	{
		randomDataInitialization();
	}

	void resultCalculation() {
		serialGaussianElimination();
		serialBackSubstitution();
	}

private:
	void randomDataInitialization() {
		for (int i = 0; i < mVector.size(); i++) {
			mVector[i] = RandomGeneration::randDoubleValue<-1000, 1000>();
			for (int j = 0; j < mVector.size(); j++) {
				mMatrix[i * mVector.size() + j] = j <= i ? RandomGeneration::randDoubleValue<-1000, 1000>() : 0;
			}
		}
	}

	void serialColumnElimination(int iteration) {
		auto pivotValue = mMatrix[iteration * mVector.size() + iteration];
		for (int i = iteration + 1; i < mVector.size(); i++) {
			double pivotFactor = mMatrix[i * mVector.size() + iteration] / pivotValue;
			for (int j = iteration; j < mVector.size(); j++) {
				mMatrix[i * mVector.size() + j] -= pivotFactor * mMatrix[iteration * mVector.size() + j];
			}
			mVector[i] -= pivotFactor * mVector[iteration];
		}
	}

	void serialGaussianElimination() {
		for (int iteration = 0; iteration < mVector.size(); iteration++) {
			serialColumnElimination(iteration);
		}
	}

	void serialBackSubstitution() {
		for (int i = mVector.size() - 1; i >= 0; i--) {
			mResult[i] = mVector[i] / mMatrix[mVector.size() * i + i];
			for (int j = 0; j < i; j++) {
				mVector[j] -= mMatrix[j * mVector.size() + i] * mResult[i];
				mMatrix[j * mVector.size() + i] = 0;
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
	SerialGauss gauss(size);

	auto start = clock();
	gauss.resultCalculation();
	auto finish = clock();
	auto duration = (finish - start) / double(CLOCKS_PER_SEC);

	ostrm << "Size = " << size << ", Time = " << duration << std::endl;
}

void main()
{
	std::string fileName = "SerialGauss";
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
	for (int i = 500; i <= 3000; i = i + 500) {
		experiment(ostrm, i);
	}
}