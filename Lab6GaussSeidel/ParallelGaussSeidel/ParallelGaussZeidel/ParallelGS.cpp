#include <optional>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>
#include <algorithm>
#include <iostream>
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
	double randDoubleValue() {
		double f = (double)rand() / RAND_MAX;
		return lowerBound + f * (upperBound - lowerBound);
	}
};

struct MPIData {
	int mProcNum = 0;
	int mProcRank = 0;
};

class ParallelGaussSeidel {
public:
	using HelperStorage = std::vector<int>;
	using DataStorage = std::vector<int>;
	using RootDataStorage = std::optional<DataStorage>;
public:
	ParallelGaussSeidel(MPIData mpiData, int size, double eps = 1) :
		mMpiData(mpiData),
		mSize(size),
		mEps(eps)
	{
		int restRows = mSize;
		for (int i = 0; i < mMpiData.mProcRank; i++) {
			restRows = restRows - restRows / (mMpiData.mProcNum - i);
		}
		mRowNum = (restRows - 2) / (mMpiData.mProcNum - mMpiData.mProcRank) + 2;

		mProcRows.resize(mRowNum * mSize);

		if (mpiData.mProcRank == 0) {
			mMatrix = DataStorage(mSize * mSize);
			randomDataInitialization();
		}
	}

	const RootDataStorage& getMatrix() const {
		return mMatrix;
	}

	int getSize() const {
		return mSize;
	}

	double getEps() const {
		return mEps;
	}

	void dataDistribution() {
		HelperStorage sendInd(mMpiData.mProcNum);
		HelperStorage sendNum(mMpiData.mProcNum);
		int restRows = mSize;
		int rowNum = (mSize - 2) / mMpiData.mProcNum + 2;
		sendNum[0] = rowNum * mSize;
		sendInd[0] = 0;
		for (int i = 1; i < mMpiData.mProcNum; i++) {
			restRows = restRows - rowNum + 1;
			rowNum = (restRows - 2) / (mMpiData.mProcNum - i) + 2;
			sendNum[i] = rowNum * mSize;
			sendInd[i] = sendInd[i - 1] + sendNum[i - 1] - mSize;
		}
		MPI_Scatterv(mMatrix ? mMatrix->data() : nullptr, sendNum.data(), sendInd.data(), MPI_INT, mProcRows.data(), sendNum[mMpiData.mProcRank], MPI_INT, 0, MPI_COMM_WORLD);
	}

	int parallelResultCalculation() {
		double dmax = 0;
		int iterations = 0;
		do {
			iterations++;
			exchangeData();
			auto dm = iterationCalculation();
			MPI_Allreduce(&dm, &dmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		} while (dmax > mEps);
		return iterations;
	}

	void resultCollection() {
		int restRows = mSize;
		HelperStorage receiveNum(mMpiData.mProcNum);
		HelperStorage receiveInd(mMpiData.mProcNum);
		receiveInd[0] = 0;
		int rowNum = (mSize - 2) / mMpiData.mProcNum + 2;
		receiveNum[0] = rowNum * mSize;
		for (int i = 1; i < mMpiData.mProcNum; i++) {
			restRows = restRows - rowNum + 1;
			rowNum = (restRows - 2) / (mMpiData.mProcNum - i) + 2;
			receiveNum[i] = rowNum * mSize;
			receiveInd[i] = receiveInd[i - 1] + receiveNum[i - 1] - mSize;
		}

		MPI_Gatherv(mProcRows.data(), receiveNum[mMpiData.mProcRank], MPI_INT, mMatrix ? mMatrix->data() : nullptr, receiveNum.data(), receiveInd.data(), MPI_INT, 0, MPI_COMM_WORLD);
	}

	static int serialResultCalculation(DataStorage& serialMatrix, int size, double eps) { 
		double dmax = 0;
		int iterations = 0;
		do {
			dmax = 0;
			for (int i = 1; i < size - 1; i++) {
				for (int j = 1; j < size - 1; j++) {
					auto temp = serialMatrix[size * i + j];
					serialMatrix[size * i + j] = 0.25 * (serialMatrix[size * i + j + 1] +
														  serialMatrix[size * i + j - 1] +
														  serialMatrix[size * (i + 1) + j] +
														  serialMatrix[size * (i - 1) + j]
														  );
					auto dm = std::fabs(serialMatrix[size * i + j] - temp);
					dmax = std::max(dmax, dm);
				}
			}
			iterations++;
		} while (dmax > eps);
		return iterations;
	}

private:
	void randomDataInitialization() {
		for (int i = 0; i < mSize; i++) {
			for (int j = 0; j < mSize; j++) {
				if (i == 0 || i == mSize - 1 || j == 0 || j == mSize - 1) {
					(*mMatrix)[i * mSize + j] = 100;
				}
				else {
					(*mMatrix)[i * mSize + j] = RandomGeneration::randDoubleValue<0, 28>();
				}
			}
		}
	}

	double iterationCalculation() {
		double dmax = 0;
		for (int i = 1; i < mRowNum - 1; i++) {
			for (int j = 1; j < mSize - 1; j++) {
				int temp = mProcRows[mSize * i + j];
				mProcRows[mSize * i + j] = 0.25 * (mProcRows[mSize * i + j + 1] +
					mProcRows[mSize * i + j - 1] +
					mProcRows[mSize * (i + 1) + j] +
					mProcRows[mSize * (i - 1) + j]
					);
				auto dm = std::fabs(mProcRows[mSize * i + j] - temp);
				dmax = std::max(dmax, dm);
			}
		}
		return dmax;
	}

	void exchangeData()
	{
		MPI_Status status;
		int nextProcNum = mMpiData.mProcRank == mMpiData.mProcNum - 1 ? MPI_PROC_NULL : mMpiData.mProcRank + 1;
		int prevProcNum = mMpiData.mProcRank == 0 ? MPI_PROC_NULL : mMpiData.mProcRank - 1;
		MPI_Sendrecv(mProcRows.data() + mSize * (mRowNum - 2), mSize, MPI_INT, nextProcNum, 4, mProcRows.data(), mSize, MPI_INT, prevProcNum, 4, MPI_COMM_WORLD, &status);
		MPI_Sendrecv(mProcRows.data() + mSize, mSize, MPI_INT, prevProcNum, 5, mProcRows.data() + (mRowNum - 1) * mSize, mSize, MPI_INT, nextProcNum, 5, MPI_COMM_WORLD, &status);
	}
private:
	MPIData mMpiData;
	int mSize;
	double mEps;
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

		std::string fileName = "ParallelGaussSeidel";

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
		experiment(100);
		for (int i = 1000; i <= 10000; i += 1000) {
			experiment(i);
		}
	}

private:
	void experiment(int size) {
		size = size / mMpiData.mProcNum * mMpiData.mProcNum + 2;
		ParallelGaussSeidel gaussSeidel(mMpiData, size);
		//ParallelGaussSeidel::DataStorage matrixCopy;
		//if (mMpiData.mProcRank == 0) {
		//	matrixCopy = *gaussSeidel.getMatrix();
		//}

		auto start = clock();
		gaussSeidel.dataDistribution();
		auto iterations = gaussSeidel.parallelResultCalculation();
		gaussSeidel.resultCollection(); 
		auto finish = clock();
		auto duration = (finish - start) / double(CLOCKS_PER_SEC);
		if (mMpiData.mProcRank == 0) {
			// testResult(gaussSeidel, matrixCopy, 1.0);
			*mOfstr << "Size = " << size << ": Iters: " << iterations << ", Time = " << duration << std::endl;
		}
	}

	void testResult(const ParallelGaussSeidel& gaussSeidel, ParallelGaussSeidel::DataStorage& matrixCopy, double eps) {
		bool equal = true;

		ParallelGaussSeidel::serialResultCalculation(matrixCopy, gaussSeidel.getSize(), gaussSeidel.getEps());
		const auto& result = gaussSeidel.getMatrix();
		for (size_t i = 0; i < matrixCopy.size(); i++) {
			if (fabs(matrixCopy[i] - result->at(i)) >= eps) {
				equal = false;
				break;
			}
		}
		if (!equal) {
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