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

class SerialSort {
public:
	SerialSort(int size)
		: mData(size)
	{
		randomDataInitialization();
	}

	void serialBubbleSort() {
		for (auto i = 1; i < mData.size(); i++) {
			for (auto j = 0; j < mData.size() - i; j++) {
				if (mData[j] > mData[j + 1]) {
					std::swap(mData[j], mData[j + 1]);
				}
			}
		}
	}

	void stdSort() {
		std::sort(mData.begin(), mData.end());
	}

	void randomDataInitialization() {
		for (auto i = 0; i < mData.size(); i++) {
			mData[i] = RandomGeneration::randDoubleValue<-100000, 100000>();
		}
	}

private:
	using DataStorage = std::vector<double>;
private:
	DataStorage mData;
};

void experiment(std::ofstream& ostrm, int size) {
	SerialSort sorter(size);

	auto start = clock();
	sorter.serialBubbleSort();
	auto finish = clock();
	auto duration = (finish - start) / double(CLOCKS_PER_SEC);
	ostrm << "Algorithm: Bubble, Size = " << size << ", Time = " << duration << std::endl;

	sorter.randomDataInitialization();
	
	start = clock();
	sorter.stdSort();
	finish = clock();
	duration = (finish - start) / double(CLOCKS_PER_SEC);
	ostrm << "Algorithm: std::sort, Size = " << size << ", Time = " << duration << std::endl;
}

void main()
{
	std::string fileName = "SerialBubble";
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
	experiment(ostrm, 1000);
	for (int i = 10000; i <= 50000; i += 10000) {
		experiment(ostrm, i);
	}
}