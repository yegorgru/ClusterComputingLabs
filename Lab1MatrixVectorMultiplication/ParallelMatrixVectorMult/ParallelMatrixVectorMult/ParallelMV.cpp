#include <ctime>
#include <fstream>
#include <string>
#include <optional>
#include <vector>
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

class ParallelMatrixVectorMult {
public:
	using DataStorage = std::vector<double>;
	using MatrixDataStorage = std::optional<DataStorage>;
private:
	using IndexesType = std::vector<int>;
	using ElementsNumberType = std::vector<int>;
public:
	ParallelMatrixVectorMult(MPIData mpiData, int size) :
		mMpiData(mpiData), 
		mSize(size)
	{
		int restRows = mSize;
		for (int i = 0; i < mpiData.mProcRank; i++) {
			restRows = restRows - restRows / (mpiData.mProcNum - i);
		}
		mRowNum = restRows / (mpiData.mProcNum - mpiData.mProcRank);

		mVector.resize(size);
		mResult.resize(size);
		mProcRows.resize(mRowNum * size);
		mProcResult.resize(mRowNum);

		if (mpiData.mProcRank == 0) {
			mMatrix = DataStorage(size * size);
			randomDataInitialization();
		}
	}

	const DataStorage& getResult() const {
		return mResult;
	}

	void dataDistribution() {
		int restRows = mSize;

		MPI_Bcast(mVector.data(), mSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		IndexesType sendInd(mMpiData.mProcNum);
		ElementsNumberType sendNum(mMpiData.mProcNum);

		auto rowNum = (mSize / (mMpiData.mProcNum));
		sendNum[0] = rowNum * mSize;
		sendInd[0] = 0;
		for (int i = 1; i < mMpiData.mProcNum; i++) {
			restRows -= rowNum;
			rowNum = restRows / (mMpiData.mProcNum - i);
			sendNum[i] = rowNum * mSize;
			sendInd[i] = sendInd[i - 1] + sendNum[i - 1];
		}

		MPI_Scatterv(mMatrix ? (*mMatrix).data() : nullptr, sendNum.data(), sendInd.data(), MPI_DOUBLE, mProcRows.data(), sendNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	void parallelResultCalculation() {
		for (int i = 0; i < mRowNum; i++) {
			mProcResult[i] = 0;
			for (int j = 0; j < mSize; j++) {
				mProcResult[i] += mProcRows[i * mSize + j] * mVector[j];
			}
		}
	}

	DataStorage serialResultCalculation() const {
		DataStorage result(mSize);
		for (int i = 0; i < mSize; i++) {
			result[i] = 0;
			for (int j = 0; j < mSize; j++) {
				result[i] += (*mMatrix)[i * mSize + j] * mVector[j];
			}
		}
		return result;
	}

	void resultReplication() {
		ElementsNumberType receiveNum(mMpiData.mProcNum);
		IndexesType receiveInd(mMpiData.mProcNum);

		receiveInd[0] = 0;
		receiveNum[0] = mSize / mMpiData.mProcNum;

		int restRows = mSize;
		for (int i = 1; i < mMpiData.mProcNum; i++) {
			restRows -= receiveNum[i - 1];
			receiveNum[i] = restRows / (mMpiData.mProcNum - i);
			receiveInd[i] = receiveInd[i - 1] + receiveNum[i - 1];
		}

		MPI_Allgatherv(mProcResult.data(), receiveNum[mMpiData.mProcRank], MPI_DOUBLE, mResult.data(), receiveNum.data(), receiveInd.data(), MPI_DOUBLE, MPI_COMM_WORLD);
	}

private:
	void randomDataInitialization() {
		auto size = mVector.size();
		for (size_t i = 0; i < size; i++) {
			mVector[i] = RandomGeneration::randDoubleValue<-1000, 1000>();
			for (size_t j = 0; j < size; j++) {
				(*mMatrix)[i * size + j] = RandomGeneration::randDoubleValue<-1000, 1000>();
			}
		}
	}

private:
	MPIData mMpiData;
	int mSize;
	MatrixDataStorage mMatrix;
	DataStorage mVector;
	DataStorage mResult;
	DataStorage mProcRows;
	DataStorage mProcResult;
	int mRowNum;
};

class Application {
public:
	Application(int argc, char* argv[]) {
		RandomGeneration::init();
		MPI_Init(&argc, &argv);

		MPI_Comm_size(MPI_COMM_WORLD, &mMpiData.mProcNum);
		MPI_Comm_rank(MPI_COMM_WORLD, &mMpiData.mProcRank);

		std::string fileName = "ParallelMV";

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
		for (int i = 1000; i <= 10000; i = i + 1000) {
			experiment(i);
		}
	}

private:
	void experiment(int size) {
		ParallelMatrixVectorMult mult(mMpiData, size);

		//auto start = MPI_Wtime();
		auto start = clock();

		mult.dataDistribution();
		mult.parallelResultCalculation();
		mult.resultReplication();

		//auto finish = MPI_Wtime();
		auto finish = clock();

		auto duration = (finish - start) / double(CLOCKS_PER_SEC);

		if (mMpiData.mProcRank == 0) {
			testResult(mult);
			*mOfstr << "Size = " << size << ", Time = " << duration << std::endl;
		}
	}

	void testResult(const ParallelMatrixVectorMult& mult) {
		bool equal = true;
		auto serialRes = mult.serialResultCalculation();
		const auto& res = mult.getResult();
		for (int i = 0; i < serialRes.size(); i++) {
			if (res[i] != serialRes[i]) {
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

void main(int argc, char* argv[])
{
	Application app(argc, argv);
	app.run();
}