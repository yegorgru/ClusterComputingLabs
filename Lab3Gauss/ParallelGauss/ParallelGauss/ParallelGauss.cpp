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
        RowNum = RestRows / (mMpiData.mProcNum - mMpiData.mProcRank);

        pProcRows.resize(RowNum * mSize);
        pProcVector.resize(RowNum);
        pProcResult.resize(RowNum);

        pParallelPivotPos.resize(mSize);
        pProcPivotIter.resize(RowNum, -1);

        pProcInd.resize(mMpiData.mProcNum);
        pProcNum.resize(mMpiData.mProcNum);

        if (mMpiData.mProcRank == 0) {
            pMatrix = DataStorage(mSize * mSize);
            pVector = DataStorage(mSize);
            pResult = DataStorage(mSize);
            RandomDataInitialization();
        }
    }

    void RandomDataInitialization() {
        for (int i = 0; i < mSize; i++) {
            (*pVector)[i] = rand() / double(1000);
            for (int j = 0; j < mSize; j++) {
                (*pMatrix)[i * mSize + j] = j <= i ? RandomGeneration::randDoubleValue<-1000, 1000>() : 0;
            }
        }
    }

    void DataDistribution() {
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
        MPI_Scatterv(pMatrix ? pMatrix->data() : nullptr, pSendNum.data(), pSendInd.data(), MPI_DOUBLE, pProcRows.data(), pSendNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Define the disposition of the matrix rows for current process
        restRows = mSize;
        pProcInd[0] = 0;
        pProcNum[0] = mSize / mMpiData.mProcNum;
        for (int i = 1; i < mMpiData.mProcNum; i++) {
            restRows -= pProcNum[i - 1];
            pProcNum[i] = restRows / (mMpiData.mProcNum - i);
            pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
        }

        MPI_Scatterv(pVector ? pVector->data() : nullptr, pProcNum.data(), pProcInd.data(), MPI_DOUBLE, pProcVector.data(), pProcNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void ResultCollection() {
        MPI_Gatherv(pProcResult.data(), pProcNum[mMpiData.mProcRank], MPI_DOUBLE, pResult ? pResult->data() : nullptr, pProcNum.data(), pProcInd.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void ParallelEliminateColumns(const DataStorage& pPivotRow, int Iter) {
        double multiplier;
        for (int i = 0; i < RowNum; i++) {
            if (pProcPivotIter[i] == -1) {
                multiplier = pProcRows[i * mSize + Iter] / pPivotRow[Iter];
                for (int j = Iter; j < mSize; j++) {
                    pProcRows[i * mSize + j] -= pPivotRow[j] * multiplier;
                }
                pProcVector[i] -= pPivotRow[mSize] * multiplier;
            }
        }
    }

    void ParallelGaussianElimination() {
        double MaxValue;
        int    PivotPos;
        struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;

        DataStorage pPivotRow(mSize + 1);

        for (int i = 0; i < mSize; i++) {
            double MaxValue = 0;
            for (int j = 0; j < RowNum; j++) {
                if ((pProcPivotIter[j] == -1) && (MaxValue < fabs(pProcRows[j * mSize + i]))) {
                    MaxValue = fabs(pProcRows[j * mSize + i]);
                    PivotPos = j;
                }
            }
            ProcPivot.MaxValue = MaxValue;
            ProcPivot.ProcRank = mMpiData.mProcRank;

            MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                MPI_COMM_WORLD);

            if (mMpiData.mProcRank == Pivot.ProcRank) {
                pProcPivotIter[PivotPos] = i;
                pParallelPivotPos[i] = pProcInd[mMpiData.mProcRank] + PivotPos;
            }
            MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.ProcRank, MPI_COMM_WORLD);

            if (mMpiData.mProcRank == Pivot.ProcRank) {
                for (int j = 0; j < mSize; j++) {
                    pPivotRow[j] = pProcRows[PivotPos * mSize + j];
                }
                pPivotRow[mSize] = pProcVector[PivotPos];
            }
            MPI_Bcast(pPivotRow.data(), mSize + 1, MPI_DOUBLE, Pivot.ProcRank, MPI_COMM_WORLD);

            ParallelEliminateColumns(pPivotRow, i);
        }
    }

    void FindBackPivotRow(int RowIndex, int& IterProcRank, int& IterPivotPos) {
        for (int i = 0; i < mMpiData.mProcNum - 1; i++) {
            if ((pProcInd[i] <= RowIndex) && (RowIndex < pProcInd[i + 1])) {
                IterProcRank = i;
            }
        }
        if (RowIndex >= pProcInd[mMpiData.mProcNum - 1]) {
            IterProcRank = mMpiData.mProcNum - 1;
        }
        IterPivotPos = RowIndex - pProcInd[IterProcRank];
    }

    void ParallelBackSubstitution() {
        int IterProcRank;
        int IterPivotPos;
        double IterResult;
        double val;

        for (int i = mSize - 1; i >= 0; i--) {
            FindBackPivotRow(pParallelPivotPos[i], IterProcRank, IterPivotPos);

            if (mMpiData.mProcRank == IterProcRank) {
                IterResult = pProcVector[IterPivotPos] / pProcRows[IterPivotPos * mSize + i];
                pProcResult[IterPivotPos] = IterResult;
            }

            MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterProcRank, MPI_COMM_WORLD);

            for (int j = 0; j < RowNum; j++)
                if (pProcPivotIter[j] < i) {
                    val = pProcRows[j * mSize + i] * IterResult;
                    pProcVector[j] = pProcVector[j] - val;
                }
        }
    }

    void ParallelResultCalculation() {
        ParallelGaussianElimination();
        ParallelBackSubstitution();
    }

    // Function for testing the result
    void TestResult() {
        int equal = 0;
        double Accuracy = 1.e-6; // Comparison accuracy

        if (mMpiData.mProcRank == 0) {
            DataStorage pRightPartVector(mSize);
            for (int i = 0; i < mSize; i++) {
                pRightPartVector[i] = 0;
                for (int j = 0; j < mSize; j++) {
                    pRightPartVector[i] += (*pMatrix)[i * mSize + j] * (*pResult)[pParallelPivotPos[j]];
                }
            }

            for (int i = 0; i < mSize; i++) {
                if (fabs(pRightPartVector[i] - (*pVector)[i]) > Accuracy) {
                    equal = 1;
                }
            }
            if (equal == 1)
                printf("The result of the parallel Gauss algorithm is NOT correct. Check your code.");
            else
                printf("The result of the parallel Gauss algorithm is correct.");
        }
    }
public:
    MPIData mMpiData;
    int mSize;

    HelperStorage pParallelPivotPos;
    HelperStorage pProcPivotIter;
    HelperStorage pProcInd;
    HelperStorage pProcNum;

    RootDataStorage pMatrix;
    RootDataStorage pVector;
    RootDataStorage pResult;
    DataStorage pProcRows;
    DataStorage pProcVector;
    DataStorage pProcResult;

    int     RowNum;
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
        //for (int i = 500; i <= 3000; i += 500) {
        //    experiment(i);
        //}
    }

private:
    void experiment(int size) {
        ParallelGauss gauss(mMpiData, size);

        //auto start = MPI_Wtime();
        //auto start = clock();

        double start, finish, duration;

        // The execution of the parallel Gauss algorithm
        start = MPI_Wtime();

        gauss.DataDistribution();

        gauss.ParallelResultCalculation();

        gauss.ResultCollection();

        finish = MPI_Wtime();
        duration = finish - start;

        gauss.TestResult();

        // Printing the time spent by Gauss algorithm
        if (mMpiData.mProcRank == 0)
            printf("\n Time of execution: %f\n", duration);

        //auto finish = MPI_Wtime();
        //auto finish = clock();

        //auto duration = (finish - start) / double(CLOCKS_PER_SEC);

        //if (mMpiData.mProcRank == 0) {
        //    testResult(mult);
        //    *mOfstr << "Size = " << size << ", Time = " << duration << std::endl;
        //}
    }

    void testResult(const ParallelGauss& mult) {
        //bool equal = true;
        //auto serialRes = mult.serialResultCalculation();
        //const auto& res = mult.getResult();
        //for (int i = 0; i < serialRes.size(); i++) {
        //    if (res[i] != serialRes[i]) {
        //        equal = false;
        //        break;
        //    }
        //}
        //if (!equal) {
        //    *mOfstr << "The results of serial and parallel algorithms are NOT identical. Check your code." << std::endl;
        //}
    }

private:
    MPIData mMpiData;
    std::optional<std::ofstream> mOfstr;
};

void main(int argc, char* argv[]) {
    Application app(argc, argv);
    app.run();
}