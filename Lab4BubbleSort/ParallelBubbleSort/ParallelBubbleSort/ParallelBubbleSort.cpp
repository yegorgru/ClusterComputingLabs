#include <algorithm>
#include <ctime>
#include <optional>
#include <vector>
#include <fstream>
#include <string>
#include <execution>
#include <mpi.h>

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

struct MPIData {
	int mProcNum = 0;
	int mProcRank = 0;
};

class ParallelSort {
public:
	using DataStorage = std::vector<double>;
	using RootDataStorage = std::optional<DataStorage>;
	using HelperDataStorage = std::vector<int>;
public:
	enum class SplitMode {
		KeepFirstHalf,
		KeepSecondHalf
	};
public:
	ParallelSort(MPIData mpiData, int size) :
		mMpiData(mpiData),
		mSize(size)
	{

		int restData = size;
		for (int i = 0; i < mpiData.mProcRank; i++) {
			restData -= restData / (mpiData.mProcNum - i);
		}
		mBlockSize = restData / (mpiData.mProcNum - mpiData.mProcRank);

		mProcData.resize(mBlockSize);

		if (mpiData.mProcRank == 0) {
			mData = RootDataStorage(size);
			randomDataInitialization();
		}
	}

	const RootDataStorage& getData() const {
		return mData;
	}

	void randomDataInitialization() {
		for (auto i = 0; i < mData->size(); i++) {
			(*mData)[i] = RandomGeneration::randDoubleValue<-100000, 100000>();
		}
	}

	void serialBubbleSort() {
		for (auto i = 1; i < mProcData.size(); i++) {
			for (int j = 0; j < mProcData.size() - i; j++) {
				if (mProcData[j] > mProcData[j + 1]) {
					std::swap(mProcData[j], mProcData[j + 1]);
				}
			}
		}
	}

	void dataDistribution() {
		HelperDataStorage sendInd(mMpiData.mProcNum);
		HelperDataStorage sendNum(mMpiData.mProcNum);

		int restData = mSize;

		int currentSize = mSize / mMpiData.mProcNum;
		sendNum[0] = currentSize;
		sendInd[0] = 0;
		for (int i = 1; i < mMpiData.mProcNum; i++) {
			restData -= currentSize;
			currentSize = restData / (mMpiData.mProcNum - i);
			sendNum[i] = currentSize;
			sendInd[i] = sendInd[i - 1] + sendNum[i - 1];
		}

		MPI_Scatterv(mData ? mData->data() : nullptr, sendNum.data(), sendInd.data(), MPI_DOUBLE, mProcData.data(), sendNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	void dataCollection() {
		HelperDataStorage receiveNum(mMpiData.mProcNum);
		HelperDataStorage receiveInd(mMpiData.mProcNum);

		int restData = mSize;

		receiveInd[0] = 0;
		receiveNum[0] = mSize / mMpiData.mProcNum;
		for (int i = 1; i < mMpiData.mProcNum; i++) {
			restData -= receiveNum[i - 1];
			receiveNum[i] = restData / (mMpiData.mProcNum - i);
			receiveInd[i] = receiveInd[i - 1] + receiveNum[i - 1];
		}

		MPI_Gatherv(mProcData.data(), mBlockSize, MPI_DOUBLE, mData ? mData->data() : nullptr, receiveNum.data(), receiveInd.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	void parallelBubble() {
		serialBubbleSort();

		int offset = 0;
		SplitMode splitMode = SplitMode::KeepFirstHalf;

		for (int i = 0; i < mMpiData.mProcNum; i++) {
			if (i % 2 == 1) {
				if (mMpiData.mProcRank % 2 == 1) {
					offset = 1;
					splitMode = SplitMode::KeepFirstHalf;
				}
				else {
					offset = -1;
					splitMode = SplitMode::KeepSecondHalf;
				}
			}
			else {
				if (mMpiData.mProcRank % 2 == 1) {
					offset = -1;
					splitMode = SplitMode::KeepSecondHalf;
				}
				else {
					offset = 1;
					splitMode = SplitMode::KeepFirstHalf;
				}
			}

			if (mMpiData.mProcRank == mMpiData.mProcNum - 1 && offset == 1 || mMpiData.mProcRank == 0 && offset == -1) {
				continue;
			}

			MPI_Status status;

			int dualBlockSize;

			MPI_Sendrecv(&mBlockSize, 1, MPI_INT, mMpiData.mProcRank + offset, 0, &dualBlockSize, 1, MPI_INT, mMpiData.mProcRank + offset, 0, MPI_COMM_WORLD, &status);

			DataStorage dualData(dualBlockSize);
			DataStorage mergedData(mBlockSize + dualBlockSize);

			exchangeData(mMpiData.mProcRank + offset, dualData);

			std::merge(mProcData.begin(), mProcData.end(), dualData.begin(), dualData.end(), mergedData.begin());

			if (splitMode == SplitMode::KeepFirstHalf) {
				std::copy(mergedData.begin(), mergedData.begin() + mBlockSize, mProcData.begin());
			}
			else {
				std::copy(mergedData.begin() + mBlockSize, mergedData.end(), mProcData.begin());
			}
		}
	}

	void exchangeData(int dualRank, DataStorage& dualData) {
		MPI_Status status;
		MPI_Sendrecv(mProcData.data(), mBlockSize, MPI_DOUBLE, dualRank, 0, dualData.data(), dualData.size(), MPI_DOUBLE, dualRank, 0, MPI_COMM_WORLD, &status);
	}

private:
	MPIData mMpiData;
	int mSize;
	int mBlockSize;
	DataStorage mProcData;
	RootDataStorage mData;
};

class Application {
public:
	Application(int argc, char* argv[]) {
		RandomGeneration::init();
		MPI_Init(&argc, &argv);

		MPI_Comm_size(MPI_COMM_WORLD, &mMpiData.mProcNum);
		MPI_Comm_rank(MPI_COMM_WORLD, &mMpiData.mProcRank);

		std::string fileName = "ParallelBubbleSort";

#ifdef NDEBUG
		fileName += "Release";
#else
		fileName += "Debug";
#endif

		fileName += std::to_string(mMpiData.mProcNum) + ".txt";

		if (mMpiData.mProcRank == 0) {
			mOfstr = std::ofstream(fileName, std::ios::out);
			*mOfstr << "Number of processes: " << mMpiData.mProcNum << std::endl;
		}
	}

	~Application() {
		MPI_Finalize();
	}

	void run() {
		experiment(10);
		experiment(100);
		experiment(1000);
		for (int i = 10000; i <= 50000; i += 10000) {
			experiment(i);
		}
	}

private:
	void experiment(int size) {
		size = size / mMpiData.mProcNum * mMpiData.mProcNum;
		ParallelSort sorter(mMpiData, size);
		ParallelSort::DataStorage copyData;
		if (mMpiData.mProcRank == 0) {
			copyData = *sorter.getData();
		}

		auto start = clock();

		sorter.dataDistribution();
		sorter.parallelBubble();
		sorter.dataCollection();

		auto finish = clock();
		auto duration = (finish - start) / double(CLOCKS_PER_SEC);

		if (mMpiData.mProcRank == 0) {
			auto stdStart = clock();
			std::sort(std::execution::par_unseq, copyData.begin(), copyData.end());
			auto stdFinish = clock();
			auto stdDuration = (stdFinish - stdStart) / double(CLOCKS_PER_SEC);
			testResult(sorter, copyData);
			*mOfstr << "Algorithm: Bubble, Size = " << size << ", Time = " << duration << std::endl;
			*mOfstr << "Algorithm: Std, Size = " << size << ", Time = " << stdDuration << std::endl;
		}
	}

	void testResult(const ParallelSort& sorter, const ParallelSort::DataStorage& sortedData) {
		const auto& dataToCheck = sorter.getData();
		if (!std::equal(dataToCheck->begin(), dataToCheck->end(), sortedData.begin())) {
			*mOfstr << "The results of serial and parallel algorithms are NOT identical. Check your code." << std::endl;
		}
	}

private:
	MPIData mMpiData;
	std::optional<std::ofstream> mOfstr;
};

void main(int argc, char* argv[]) {
	Application app(argc, argv);
	app.run();
}