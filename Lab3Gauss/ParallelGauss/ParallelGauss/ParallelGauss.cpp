#include <ctime>
#include <cmath>
#include <optional>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
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

class ParallelGauss {
public:
    using HelperStorage = std::vector<int>;
    using DataStorage = std::vector<double>;
    using RootDataStorage = std::optional<DataStorage>;
public:
    ParallelGauss(MPIData mpiData, int size) :
        mMpiData(mpiData),
        mSize(size)
    {
        int RestRows = mSize;
        for (int i = 0; i < mMpiData.mProcRank; i++) {
            RestRows = RestRows - RestRows / (mMpiData.mProcNum - i);
        }
        mRowNum = RestRows / (mMpiData.mProcNum - mMpiData.mProcRank);

        mProcRows.resize(mRowNum * mSize);
        mProcVector.resize(mRowNum);
        mProcResult.resize(mRowNum);

        mParallelPivotPos.resize(mSize);
        mProcPivotIter.resize(mRowNum, -1);

        mProcInd.resize(mMpiData.mProcNum);
        mProcNum.resize(mMpiData.mProcNum);

        if (mMpiData.mProcRank == 0) {
            mMatrix = DataStorage(mSize * mSize);
            mVector = DataStorage(mSize);
            mResult = DataStorage(mSize);
            randomDataInitialization();
        }
    }

    const RootDataStorage& getMatrix() const {
        return mMatrix;
    }

    const RootDataStorage& getVector() const {
        return mVector;
    }

    const RootDataStorage& getResult() const {
        return mResult;
    }

    const HelperStorage& getParallelPivotPos() const {
        return mParallelPivotPos;
    }

    void dataDistribution() {
        HelperStorage pSendNum(mMpiData.mProcNum);
        HelperStorage pSendInd(mMpiData.mProcNum);

        int restRows = mSize;

        int rowNum = (mSize / mMpiData.mProcNum);
        pSendNum[0] = rowNum * mSize;
        pSendInd[0] = 0;
        for (int i = 1; i < mMpiData.mProcNum; i++) {
            restRows -= rowNum;
            rowNum = restRows / (mMpiData.mProcNum - i);
            pSendNum[i] = rowNum * mSize;
            pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
        }

        // Scatter the rows
        MPI_Scatterv(mMatrix ? mMatrix->data() : nullptr, pSendNum.data(), pSendInd.data(), MPI_DOUBLE, mProcRows.data(), pSendNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Define the disposition of the matrix rows for current process
        restRows = mSize;
        mProcInd[0] = 0;
        mProcNum[0] = mSize / mMpiData.mProcNum;
        for (int i = 1; i < mMpiData.mProcNum; i++) {
            restRows -= mProcNum[i - 1];
            mProcNum[i] = restRows / (mMpiData.mProcNum - i);
            mProcInd[i] = mProcInd[i - 1] + mProcNum[i - 1];
        }

        MPI_Scatterv(mVector ? mVector->data() : nullptr, mProcNum.data(), mProcInd.data(), MPI_DOUBLE, mProcVector.data(), mProcNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void resultCollection() {
        MPI_Gatherv(mProcResult.data(), mProcNum[mMpiData.mProcRank], MPI_DOUBLE, mResult ? mResult->data() : nullptr, mProcNum.data(), mProcInd.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void parallelResultCalculation() {
        parallelGaussianElimination();
        parallelBackSubstitution();
    }

private:
    void randomDataInitialization() {
        for (int i = 0; i < mSize; i++) {
            (*mVector)[i] = rand() / double(1000);
            for (int j = 0; j < mSize; j++) {
                (*mMatrix)[i * mSize + j] = j <= i ? RandomGeneration::randDoubleValue<-1000, 1000>() : 0;
            }
        }
    }

    void parallelGaussianElimination() {
        double MaxValue;
        int    PivotPos;
        struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;

        DataStorage pPivotRow(mSize + 1);

        for (int i = 0; i < mSize; i++) {
            double MaxValue = 0;
            for (int j = 0; j < mRowNum; j++) {
                if ((mProcPivotIter[j] == -1) && (MaxValue < fabs(mProcRows[j * mSize + i]))) {
                    MaxValue = fabs(mProcRows[j * mSize + i]);
                    PivotPos = j;
                }
            }
            ProcPivot.MaxValue = MaxValue;
            ProcPivot.ProcRank = mMpiData.mProcRank;

            MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                MPI_COMM_WORLD);

            if (mMpiData.mProcRank == Pivot.ProcRank) {
                mProcPivotIter[PivotPos] = i;
                mParallelPivotPos[i] = mProcInd[mMpiData.mProcRank] + PivotPos;
            }
            MPI_Bcast(&mParallelPivotPos[i], 1, MPI_INT, Pivot.ProcRank, MPI_COMM_WORLD);

            if (mMpiData.mProcRank == Pivot.ProcRank) {
                for (int j = 0; j < mSize; j++) {
                    pPivotRow[j] = mProcRows[PivotPos * mSize + j];
                }
                pPivotRow[mSize] = mProcVector[PivotPos];
            }
            MPI_Bcast(pPivotRow.data(), mSize + 1, MPI_DOUBLE, Pivot.ProcRank, MPI_COMM_WORLD);

            parallelEliminateColumns(pPivotRow, i);
        }
    }

    void parallelBackSubstitution() {
        int IterProcRank;
        int IterPivotPos;
        double IterResult;
        double val;

        for (int i = mSize - 1; i >= 0; i--) {
            findBackPivotRow(mParallelPivotPos[i], IterProcRank, IterPivotPos);

            if (mMpiData.mProcRank == IterProcRank) {
                IterResult = mProcVector[IterPivotPos] / mProcRows[IterPivotPos * mSize + i];
                mProcResult[IterPivotPos] = IterResult;
            }

            MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterProcRank, MPI_COMM_WORLD);

            for (int j = 0; j < mRowNum; j++)
                if (mProcPivotIter[j] < i) {
                    val = mProcRows[j * mSize + i] * IterResult;
                    mProcVector[j] = mProcVector[j] - val;
                }
        }
    }

    void findBackPivotRow(int RowIndex, int& IterProcRank, int& IterPivotPos) {
        for (int i = 0; i < mMpiData.mProcNum - 1; i++) {
            if ((mProcInd[i] <= RowIndex) && (RowIndex < mProcInd[i + 1])) {
                IterProcRank = i;
            }
        }
        if (RowIndex >= mProcInd[mMpiData.mProcNum - 1]) {
            IterProcRank = mMpiData.mProcNum - 1;
        }
        IterPivotPos = RowIndex - mProcInd[IterProcRank];
    }

    void parallelEliminateColumns(const DataStorage& pPivotRow, int Iter) {
        double multiplier;
        for (int i = 0; i < mRowNum; i++) {
            if (mProcPivotIter[i] == -1) {
                multiplier = mProcRows[i * mSize + Iter] / pPivotRow[Iter];
                for (int j = Iter; j < mSize; j++) {
                    mProcRows[i * mSize + j] -= pPivotRow[j] * multiplier;
                }
                mProcVector[i] -= pPivotRow[mSize] * multiplier;
            }
        }
    }
public:
    MPIData mMpiData;
    int mSize;

    HelperStorage mParallelPivotPos;
    HelperStorage mProcPivotIter;
    HelperStorage mProcInd;
    HelperStorage mProcNum;

    RootDataStorage mMatrix;
    RootDataStorage mVector;
    RootDataStorage mResult;
    DataStorage mProcRows;
    DataStorage mProcVector;
    DataStorage mProcResult;

    int     mRowNum;
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
        /*for (int i = 500; i <= 3000; i += 500) {
            experiment(i);
        }*/
    }

private:
    void experiment(int size) {
        ParallelGauss gauss(mMpiData, size);

        auto start = clock();

        gauss.dataDistribution();

        gauss.parallelResultCalculation();

        gauss.resultCollection();

        auto finish = clock();
        auto duration = (finish - start) / double(CLOCKS_PER_SEC);

        if (mMpiData.mProcRank == 0) {
            testResult(gauss, size);
            *mOfstr << "Size = " << size << ", Time = " << duration << std::endl;
        }
    }

    void testResult(const ParallelGauss& gauss, int size) {
        bool equal = true;
        double accuracy = 1.e-6;

        const auto& matrix = gauss.getMatrix();
        const auto& vec = gauss.getVector();
        const auto& result = gauss.getResult();
        const auto& parallelPivotPos = gauss.getParallelPivotPos();

        ParallelGauss::DataStorage rightPartVector(size);
        for (int i = 0; i < size; i++) {
            rightPartVector[i] = 0;
            for (int j = 0; j < size; j++) {
                rightPartVector[i] += (*matrix)[i * size + j] * (*result)[parallelPivotPos[j]];
            }
        }

        for (int i = 0; i < size; i++) {
            if (fabs(rightPartVector[i] - (*vec)[i]) > accuracy) {
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