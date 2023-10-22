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
    }

    void RandomDataInitialization(double* pMatrix, double* pVector) {
        for (int i = 0; i < mSize; i++) {
            pVector[i] = rand() / double(1000);
            for (int j = 0; j < mSize; j++) {
                if (j <= i)
                    pMatrix[i * mSize + j] = rand() / double(1000);
                else
                    pMatrix[i * mSize + j] = 0;
            }
        }
    }

    void ProcessInitialization(double*& pMatrix, double*& pVector, double*& pResult, int& RowNum) {

        int RestRows; // Number of rows, that haven't been distributed yet

        RestRows = mSize;
        for (int i = 0; i < mMpiData.mProcRank; i++) {
            RestRows = RestRows - RestRows / (mMpiData.mProcNum - i);
        }
        RowNum = RestRows / (mMpiData.mProcNum - mMpiData.mProcRank);

        pProcRows.resize(RowNum * mSize);
        pProcVector.resize(RowNum);
        pProcResult.resize(RowNum);

        pParallelPivotPos.resize(mSize);
        pProcPivotIter.resize(RowNum);

        pProcInd.resize(mMpiData.mProcNum);
        pProcNum.resize(mMpiData.mProcNum);

        for (int i = 0; i < RowNum; i++)
            pProcPivotIter[i] = -1;

        if (mMpiData.mProcRank == 0) {
            pMatrix = new double[mSize * mSize];
            pVector = new double[mSize];
            pResult = new double[mSize];
            RandomDataInitialization(pMatrix, pVector);
        }
    }

    void DataDistribution(double* pMatrix, double* pVector, int RowNum) {

        HelperStorage pSendNum(mMpiData.mProcNum);
        HelperStorage pSendInd(mMpiData.mProcNum);
        int RestRows = mSize;

        RowNum = (mSize / mMpiData.mProcNum);
        pSendNum[0] = RowNum * mSize;
        pSendInd[0] = 0;
        for (int i = 1; i < mMpiData.mProcNum; i++) {
            RestRows -= RowNum;
            RowNum = RestRows / (mMpiData.mProcNum - i);
            pSendNum[i] = RowNum * mSize;
            pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
        }

        // Scatter the rows
        MPI_Scatterv(pMatrix, pSendNum.data(), pSendInd.data(), MPI_DOUBLE, pProcRows.data(), pSendNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Define the disposition of the matrix rows for current process
        RestRows = mSize;
        pProcInd[0] = 0;
        pProcNum[0] = mSize / mMpiData.mProcNum;
        for (int i = 1; i < mMpiData.mProcNum; i++) {
            RestRows -= pProcNum[i - 1];
            pProcNum[i] = RestRows / (mMpiData.mProcNum - i);
            pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
        }

        MPI_Scatterv(pVector, pProcNum.data(), pProcInd.data(), MPI_DOUBLE, pProcVector.data(), pProcNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void ResultCollection(double* pResult) {
        MPI_Gatherv(pProcResult.data(), pProcNum[mMpiData.mProcRank], MPI_DOUBLE, pResult, pProcNum.data(), pProcInd.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    void ParallelEliminateColumns(const DataStorage& pPivotRow, int RowNum, int Iter) {
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


    // Function for the Gausian elimination
    void ParallelGaussianElimination(int RowNum) {
        double MaxValue;   // Value of the pivot element of th  process
        int    PivotPos;   // Position of the pivot row in the process stripe 
        // Structure for the pivot row selection
        struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;

        // pPivotRow is used for storing the pivot row and the corresponding 
        // element of the vector b
        DataStorage pPivotRow(mSize + 1);

        // The iterations of the Gaussian elimination stage
        for (int i = 0; i < mSize; i++) {

            // Calculating the local pivot row
            double MaxValue = 0;
            for (int j = 0; j < RowNum; j++) {
                if ((pProcPivotIter[j] == -1) && (MaxValue < fabs(pProcRows[j * mSize + i]))) {
                    MaxValue = fabs(pProcRows[j * mSize + i]);
                    PivotPos = j;
                }
            }
            ProcPivot.MaxValue = MaxValue;
            ProcPivot.ProcRank = mMpiData.mProcRank;

            // Finding the pivot process (process with the maximum value of MaxValue)
            MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                MPI_COMM_WORLD);

            // Broadcasting the pivot row
            if (mMpiData.mProcRank == Pivot.ProcRank) {
                pProcPivotIter[PivotPos] = i; //iteration number
                pParallelPivotPos[i] = pProcInd[mMpiData.mProcRank] + PivotPos;
            }
            MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.ProcRank, MPI_COMM_WORLD);

            if (mMpiData.mProcRank == Pivot.ProcRank) {
                // Fill the pivot row
                for (int j = 0; j < mSize; j++) {
                    pPivotRow[j] = pProcRows[PivotPos * mSize + j];
                }
                pPivotRow[mSize] = pProcVector[PivotPos];
            }
            MPI_Bcast(pPivotRow.data(), mSize + 1, MPI_DOUBLE, Pivot.ProcRank, MPI_COMM_WORLD);

            ParallelEliminateColumns(pPivotRow, RowNum, i);
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

    void ParallelBackSubstitution(int RowNum) {
        int IterProcRank;    // Rank of the process with the current pivot row
        int IterPivotPos;    // Position of the pivot row of the process
        double IterResult;   // Calculated value of the current unknown
        double val;

        // Iterations of the back substitution stage
        for (int i = mSize - 1; i >= 0; i--) {

            // Calculating the rank of the process, which holds the pivot row
            FindBackPivotRow(pParallelPivotPos[i], IterProcRank, IterPivotPos);

            // Calculating the unknown
            if (mMpiData.mProcRank == IterProcRank) {
                IterResult = pProcVector[IterPivotPos] / pProcRows[IterPivotPos * mSize + i];
                pProcResult[IterPivotPos] = IterResult;
            }
            // Broadcasting the value of the current unknown
            MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterProcRank, MPI_COMM_WORLD);

            // Updating the values of the vector b
            for (int j = 0; j < RowNum; j++)
                if (pProcPivotIter[j] < i) {
                    val = pProcRows[j * mSize + i] * IterResult;
                    pProcVector[j] = pProcVector[j] - val;
                }
        }
    }

    // Function for the execution of the parallel Gauss algorithm
    void ParallelResultCalculation(int RowNum) {
        ParallelGaussianElimination(RowNum);
        ParallelBackSubstitution(RowNum);
    }

    // Function for computational process termination
    void ProcessTermination(double* pMatrix, double* pVector, double* pResult) {
        if (mMpiData.mProcRank == 0) {
            delete[] pMatrix;
            delete[] pVector;
            delete[] pResult;
        }
    }


    // Function for testing the result
    void TestResult(double* pMatrix, double* pVector, double* pResult) {
        /* Buffer for storing the vector, that is a result of multiplication
           of the linear system matrix by the vector of unknowns */
        double* pRightPartVector;
        // Flag, that shows wheather the right parts vectors are identical or not
        int equal = 0;
        double Accuracy = 1.e-6; // Comparison accuracy

        if (mMpiData.mProcRank == 0) {
            pRightPartVector = new double[mSize];
            for (int i = 0; i < mSize; i++) {
                pRightPartVector[i] = 0;
                for (int j = 0; j < mSize; j++) {
                    pRightPartVector[i] += pMatrix[i * mSize + j] * pResult[pParallelPivotPos[j]];
                }
            }

            for (int i = 0; i < mSize; i++) {
                if (fabs(pRightPartVector[i] - pVector[i]) > Accuracy) {
                    equal = 1;
                }
            }
            if (equal == 1)
                printf("The result of the parallel Gauss algorithm is NOT correct. Check your code.");
            else
                printf("The result of the parallel Gauss algorithm is correct.");
            delete[] pRightPartVector;
        }
    }
public:
    MPIData mMpiData;
    int mSize;

    HelperStorage pParallelPivotPos;
    HelperStorage pProcPivotIter;
    HelperStorage pProcInd;
    HelperStorage pProcNum;

    double* pMatrix;        // Matrix of the linear system
    double* pVector;        // Right parts of the linear system
    double* pResult;        // Result vector
    DataStorage pProcRows;      // Rows of the matrix A
    DataStorage pProcVector;    // Block of the vector b
    DataStorage pProcResult;    // Block of the vector x

    int     RowNum;         // Number of  the matrix rows 
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

        // Memory allocation and data initialization
        gauss.ProcessInitialization(gauss.pMatrix, gauss.pVector, gauss.pResult, gauss.RowNum);
        // The execution of the parallel Gauss algorithm
        start = MPI_Wtime();

        gauss.DataDistribution(gauss.pMatrix, gauss.pVector, gauss.RowNum);

        gauss.ParallelResultCalculation(gauss.RowNum);

        gauss.ResultCollection(gauss.pResult);

        finish = MPI_Wtime();
        duration = finish - start;

        gauss.TestResult(gauss.pMatrix, gauss.pVector, gauss.pResult);

        // Printing the time spent by Gauss algorithm
        if (mMpiData.mProcRank == 0)
            printf("\n Time of execution: %f\n", duration);

        // Computational process termination
        gauss.ProcessTermination(gauss.pMatrix, gauss.pVector, gauss.pResult);

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