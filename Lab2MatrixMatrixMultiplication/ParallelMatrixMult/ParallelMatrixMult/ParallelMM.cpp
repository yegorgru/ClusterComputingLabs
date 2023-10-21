#include <ctime>
#include <cmath>
#include <fstream>
#include <optional>
#include <vector>
#include <string>
#include <iostream>
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

class ParallelMatrixMatrixMult {
private:
	using BlockDataStorage = std::vector<double>;
	using DataStorage = std::optional<BlockDataStorage>;
public:
	ParallelMatrixMatrixMult(MPIData mpiData, int size, int gridSize)
		: mMpiData(mpiData), mSize(size), mGridSize(gridSize), mGridCoords(2, 0), mGridComm(), mColComm(), mRowComm()
	{
		mBlockSize = mSize / mGridSize;
		mBlockA.resize(mBlockSize * mBlockSize);
		mBlockB.resize(mBlockSize * mBlockSize);
		mBlockC.resize(mBlockSize * mBlockSize, 0);
		mBlockATemp.resize(mBlockSize * mBlockSize);
		if (mMpiData.mProcRank == 0) {
			mMatrixA = DataStorage(mSize * mSize);
			mMatrixB = DataStorage(mSize * mSize);
			mMatrixC = DataStorage(mSize * mSize);
			std::fill(mMatrixC->begin(), mMatrixC->end(), 0);
			randomDataInitialization();
		}
	}

	const BlockDataStorage& getResult() const {
		return *mMatrixC;
	}

	BlockDataStorage serialResultCalculation() {
		BlockDataStorage matrixC(mSize * mSize, 0);
		serialResultCalculation(*mMatrixA, *mMatrixB, matrixC);
		return matrixC;
	}

	void createGridCommunicators() {
		std::vector<int> dimSize{ mGridSize , mGridSize };
		std::vector<int> periodic{0, 0};

		MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize.data(), periodic.data(), 1, &mGridComm);
		MPI_Cart_coords(mGridComm, mMpiData.mProcRank, 2, mGridCoords.data());

		std::vector<int> subdims{0, 1};
		MPI_Cart_sub(mGridComm, subdims.data(), &mRowComm);

		subdims = {1, 0};
		MPI_Cart_sub(mGridComm, subdims.data(), &mColComm);
	}

	void dataDistribution() {
		checkerboardMatrixScatter(mMatrixA, mBlockATemp);
		checkerboardMatrixScatter(mMatrixB, mBlockB);
	}

	void parallelResultCalculation() {
		for (int iter = 0; iter < mGridSize; iter++) {
			aBlockCommunication(iter);
			serialResultCalculation(mBlockA, mBlockB, mBlockC);
			bBlockCommunication();
		}
	}

	void resultCollection() {
		BlockDataStorage resultRow(mSize * mBlockSize);
		for (int i = 0; i < mBlockSize; i++) {
			MPI_Gather(&mBlockC[i * mBlockSize], mBlockSize, MPI_DOUBLE, &resultRow[i * mSize], mBlockSize, MPI_DOUBLE, 0, mRowComm);
		}
		if (mGridCoords[1] == 0) {
			MPI_Gather(resultRow.data(), mBlockSize * mSize, MPI_DOUBLE, mMatrixC ? mMatrixC->data() : nullptr, mBlockSize * mSize, MPI_DOUBLE, 0, mColComm);
		}
	}
private:
	void serialResultCalculation(BlockDataStorage& matrixA, BlockDataStorage& matrixB, BlockDataStorage& matrixC) {
		int size = std::sqrt(matrixA.size());
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < size; k++) {
					matrixC[i * size + j] += matrixA[i * size + k] * matrixB[k * size + j];
				}
			}
		}
	}

	void randomDataInitialization() {
		for (int i = 0; i < mSize; i++) {
			for (int j = 0; j < mSize; j++) {
				(*mMatrixA)[i * mSize + j] = RandomGeneration::randDoubleValue<-1000, 1000>();
				(*mMatrixB)[i * mSize + j] = RandomGeneration::randDoubleValue<-1000, 1000>();
			}
		}
	}

	void checkerboardMatrixScatter(DataStorage& matrix, BlockDataStorage& matrixBlock) {
		BlockDataStorage matrixRow(mBlockSize * mSize);
		if (mGridCoords[1] == 0) {
			MPI_Scatter(matrix ? matrix->data() : nullptr, mBlockSize * mSize, MPI_DOUBLE, matrixRow.data(), mBlockSize * mSize, MPI_DOUBLE, 0, mColComm);
		}
		for (int i = 0; i < mBlockSize; i++) {
			MPI_Scatter(&matrixRow[i * mSize], mBlockSize, MPI_DOUBLE, &(matrixBlock[i * mBlockSize]), mBlockSize, MPI_DOUBLE, 0, mRowComm);
		}
	}

	void aBlockCommunication(int iter) {
		int pivot = (mGridCoords[0] + iter) % mGridSize;
		if (mGridCoords[1] == pivot) {
			std::copy(mBlockATemp.begin(), mBlockATemp.begin() + mBlockSize * mBlockSize, mBlockA.begin());
		}
		MPI_Bcast(mBlockA.data(), mBlockSize * mBlockSize, MPI_DOUBLE, pivot, mRowComm);
	}

	void bBlockCommunication() {
		MPI_Status status;
		int nextProc = mGridCoords[0] + 1;
		if (mGridCoords[0] == mGridSize - 1) {
			nextProc = 0;
		}
		int prevProc = mGridCoords[0] - 1;
		if (mGridCoords[0] == 0) {
			prevProc = mGridSize - 1;
		}
		MPI_Sendrecv_replace(mBlockB.data(), mBlockSize * mBlockSize, MPI_DOUBLE, nextProc, 0, prevProc, 0, mColComm, &status);
	}
private:
	using GridCoordinatesType = std::vector<int>;
private:
	MPIData mMpiData;
	int mSize;
	int mGridSize = 0;
	int mBlockSize = 0;
	GridCoordinatesType mGridCoords;
	DataStorage mMatrixA;
	DataStorage mMatrixB;
	DataStorage mMatrixC;
	BlockDataStorage mBlockA;
	BlockDataStorage mBlockB;
	BlockDataStorage mBlockC;
	BlockDataStorage mBlockATemp;
	MPI_Comm mGridComm;
	MPI_Comm mColComm;
	MPI_Comm mRowComm;
};

class Application {
public:
	Application(int argc, char* argv[]) {
		RandomGeneration::init();
		MPI_Init(&argc, &argv);

		MPI_Comm_size(MPI_COMM_WORLD, &mMpiData.mProcNum);
		MPI_Comm_rank(MPI_COMM_WORLD, &mMpiData.mProcRank);

		std::string fileName = "ParallelMM";

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
		if (mMpiData.mProcNum != sqrt((double)mMpiData.mProcNum) * sqrt((double)mMpiData.mProcNum)) {
			if (mMpiData.mProcRank == 0) {
				* mOfstr << "Number of processes must be a perfect square" << std::endl;
				return;
			}
		}
		experiment(10);
		experiment(100);
		for (int i = 500; i <= 3000; i = i + 500) {
			experiment(i);
		}
	}

private:
	void experiment(int size) {
		int gridSize = sqrt((double)mMpiData.mProcNum);
		size = size / gridSize * gridSize;
		ParallelMatrixMatrixMult mult(mMpiData, size, gridSize);

		//auto start = MPI_Wtime();
		auto start = clock();

		mult.createGridCommunicators();
		mult.dataDistribution();
		mult.parallelResultCalculation();
		mult.resultCollection();

		//auto finish = MPI_Wtime();
		auto finish = clock();

		auto duration = (finish - start) / double(CLOCKS_PER_SEC);

		if (mMpiData.mProcRank == 0) {
			testResult(mult);
			*mOfstr << "Size = " << size << ", Time = " << duration << std::endl;
		}
	}

	void testResult(ParallelMatrixMatrixMult& mult) {
		bool equal = true;
		auto serialRes = mult.serialResultCalculation();
		const auto& res = mult.getResult();
		double accuracy = 1.e-4; // Comparison accuracy
		for (int i = 0; i < serialRes.size(); i++) {
			if (std::fabs(res[i] - serialRes[i]) > accuracy) {
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