#include <optional>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>
#include <algorithm>
#include <mpi.h>

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

struct MPIData {
	int mProcNum = 0;
	int mProcRank = 0;
};

class ParallelFloyd {
public:
	using HelperStorage = std::vector<int>;
	using DataStorage = std::vector<int>;
	using RootDataStorage = std::optional<DataStorage>;
public:
	ParallelFloyd(MPIData mpiData, int size, double infinitiesPercent = 50.0) :
		mMpiData(mpiData),
		mSize(size)
	{
		int restRows = mSize;
		for (int i = 0; i < mMpiData.mProcRank; i++) {
			restRows = restRows - restRows / (mMpiData.mProcNum - i);
		}
		mRowNum = restRows / (mMpiData.mProcNum - mMpiData.mProcRank);

		mProcRows.resize(mSize * mRowNum);

		if (mpiData.mProcRank == 0) {
			mMatrix = DataStorage(mSize * mSize);
			randomDataInitialization(infinitiesPercent);
		}
	}

	const RootDataStorage& getMatrix() const {
		return mMatrix;
	}

	void dataDistribution() {
		int restRows = mSize;

		HelperStorage sendInd(mMpiData.mProcNum);
		HelperStorage sendNum(mMpiData.mProcNum);

		int rowNum = mSize / mMpiData.mProcNum;
		sendNum[0] = rowNum * mSize;
		sendInd[0] = 0;
		for (int i = 1; i < mMpiData.mProcNum; i++) {
			restRows -= rowNum;
			rowNum = restRows / (mMpiData.mProcNum - i);
			sendNum[i] = rowNum * mSize;
			sendInd[i] = sendInd[i - 1] + sendNum[i - 1];
		}

		MPI_Scatterv(mMatrix ? mMatrix->data() : nullptr, sendNum.data(), sendInd.data(), MPI_INT, mProcRows.data(), sendNum[mMpiData.mProcRank], MPI_INT, 0, MPI_COMM_WORLD);
	}

	void parallelFloyd() {
		HelperStorage row(mSize);
		for (int k = 0; k < mSize; k++) {
			rowDistribution(k, row);
			for (int i = 0; i < mRowNum; i++) {
				for (int j = 0; j < mSize; j++) {
					if (mProcRows[i * mSize + k] != -1 && row[j] != -1) {
						mProcRows[i * mSize + j] = findMin(mProcRows[i * mSize + j], mProcRows[i * mSize + k] + row[j]);
					}
				}
			}
		}
	}

	void resultCollection() {
		int restRows = mSize;
		HelperStorage receiveNum(mMpiData.mProcNum);
		HelperStorage receiveInd(mMpiData.mProcNum);

		int rowNum = mSize / mMpiData.mProcNum;
		receiveInd[0] = 0;
		receiveNum[0] = rowNum * mSize;

		for (int i = 1; i < mMpiData.mProcNum; i++) {
			restRows -= rowNum;
			rowNum = restRows / (mMpiData.mProcNum - i);
			receiveNum[i] = rowNum * mSize;
			receiveInd[i] = receiveInd[i - 1] + receiveNum[i - 1];
		}
		MPI_Gatherv(mProcRows.data(), receiveNum[mMpiData.mProcRank], MPI_INT, mMatrix ? mMatrix->data() : nullptr, receiveNum.data(), receiveInd.data(), MPI_INT, 0, MPI_COMM_WORLD);
	}

	void serialFloyd(DataStorage& matrix) const {
		for (int k = 0; k < mSize; k++) {
			for (int i = 0; i < mSize; i++) {
				for (int j = 0; j < mSize; j++) {
					if (matrix[i * mSize + k] != -1 && matrix[k * mSize + j] != -1) {
						matrix[i * mSize + j] = findMin(matrix[i * mSize + j], matrix[i * mSize + k] + matrix[k * mSize + j]);
					}
				}
			}
		}
	}

private:
	void randomDataInitialization(double infinitiesPercent) {
		using namespace RandomGeneration;
		for (int i = 0; i < mSize; i++) {
			for (int j = 0; j < mSize; j++) {
				if (i != j) {
					(*mMatrix)[i * mSize + j] = randDoubleValue<0, 100>() < infinitiesPercent ? -1 : randIntValue<1, 1000>();
				}
				else {
					(*mMatrix)[i * mSize + j] = 0;
				}
			}
		}
	}

	int findMin(int lhs, int rhs) const {
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

	void rowDistribution(int k, HelperStorage& row) {
		int restRows = mSize;
		int ind = 0;
		int num = mSize / mMpiData.mProcNum;

		int procRowRank = 1;
		for (; procRowRank < mMpiData.mProcNum + 1; procRowRank++) {
			if (k < ind + num) {
				break;
			}
			restRows -= num;
			ind += num;
			num = restRows / (mMpiData.mProcNum - procRowRank);
		}
		--procRowRank;
		int procRowNum = k - ind;

		if (procRowRank == mMpiData.mProcRank) {
			std::copy(mProcRows.begin() + procRowNum * mSize, mProcRows.begin() + (procRowNum + 1) * mSize, row.begin());
		}

		MPI_Bcast(row.data(), mSize, MPI_INT, procRowRank, MPI_COMM_WORLD);
	}
private:
	MPIData mMpiData;
	int mSize;
	int mRowNum;
	RootDataStorage mMatrix;
	DataStorage mProcRows;
};

class Application {
public:
	Application(int argc, char* argv[]) {
		RandomGeneration::init();
		MPI_Init(&argc, &argv);

		MPI_Comm_size(MPI_COMM_WORLD, &mMpiData.mProcNum);
		MPI_Comm_rank(MPI_COMM_WORLD, &mMpiData.mProcRank);

		std::string fileName = "ParallelFloyd";

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
		for (int i = 500; i <= 1000; i += 100) {
			experiment(i);
		}
	}

private:
	void experiment(int size) {
		ParallelFloyd floyd(mMpiData, size);
		ParallelFloyd::DataStorage matrixCopy;
		if (mMpiData.mProcRank == 0) {
			matrixCopy = *floyd.getMatrix();
		}

		auto start = clock();

		floyd.dataDistribution();
		floyd.parallelFloyd();
		floyd.resultCollection();

		auto finish = clock();
		auto duration = (finish - start) / double(CLOCKS_PER_SEC);

		if (mMpiData.mProcRank == 0) {
			testResult(floyd, matrixCopy);
			*mOfstr << "Size = " << size << ", Time = " << duration << std::endl;
		}
	}

	void testResult(const ParallelFloyd& floyd, ParallelFloyd::DataStorage matrixCopy) {
		floyd.serialFloyd(matrixCopy);
		const auto& matrix = floyd.getMatrix();
		if (!std::equal(matrix->begin(), matrix->end(), matrixCopy.begin())) {
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