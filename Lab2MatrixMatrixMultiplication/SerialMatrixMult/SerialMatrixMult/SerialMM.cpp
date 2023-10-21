#include <ctime>
#include <vector>
#include <fstream>

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

class SerialMatrixMatrixMult {
public:
	SerialMatrixMatrixMult(int size)
		: mSize(size), mMatrixA(size* size), mMatrixB(size* size), mMatrixC(size* size, 0)
	{
	}

	void randomDataInitialization() {
		for (size_t i = 0; i < mMatrixA.size(); ++i) {
			mMatrixA[i] = RandomGeneration::randDoubleValue<-1000, 1000>();
			mMatrixB[i] = RandomGeneration::randDoubleValue<-1000, 1000>();
		}
	}

	void resultCalculation() {
		for (int i = 0; i < mSize; i++) {
			for (int j = 0; j < mSize; j++) {
				for (int k = 0; k < mSize; k++) {
					mMatrixC[i * mSize + j] += mMatrixA[i * mSize + k] * mMatrixB[k * mSize + j];
				}
			}
		}
	}

private:
	using DataStorage = std::vector<double>;
private:
	int mSize;
	DataStorage mMatrixA;
	DataStorage mMatrixB;
	DataStorage mMatrixC;
};

void experiment(std::ofstream& ostrm, int size) {
	SerialMatrixMatrixMult mult(size);
	mult.randomDataInitialization();

	auto start = clock();
	mult.resultCalculation();
	auto finish = clock();
	auto duration = (finish - start) / double(CLOCKS_PER_SEC);

	ostrm << "Size = " << size << ", Time = " << duration << std::endl;
}

void main()
{
	std::string fileName = "SerialMM";
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
