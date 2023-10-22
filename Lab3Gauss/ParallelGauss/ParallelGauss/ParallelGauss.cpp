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
    ParallelGauss(MPIData mpiData, int size) :
        mMpiData(mpiData),
        mSize(size)
    {
    }

    void RandomDataInitialization(double* pMatrix, double* pVector) {
        int i, j;  // Loop variables
        srand(unsigned(clock()));
        for (i = 0; i < mSize; i++) {
            pVector[i] = rand() / double(1000);
            for (j = 0; j < mSize; j++) {
                if (j <= i)
                    pMatrix[i * mSize + j] = rand() / double(1000);
                else
                    pMatrix[i * mSize + j] = 0;
            }
        }
    }

    void ProcessInitialization(double*& pMatrix, double*& pVector, double*& pResult, double*& pProcRows, double*& pProcVector, double*& pProcResult, int& RowNum) {

        int RestRows; // Number of rows, that haven't been distributed yet

        RestRows = mSize;
        for (int i = 0; i < mMpiData.mProcRank; i++) {
            RestRows = RestRows - RestRows / (mMpiData.mProcNum - i);
        }
        RowNum = RestRows / (mMpiData.mProcNum - mMpiData.mProcRank);

        pProcRows = new double[RowNum * mSize];
        pProcVector = new double[RowNum];
        pProcResult = new double[RowNum];

        pParallelPivotPos = new int[mSize];
        pProcPivotIter = new int[RowNum];

        pProcInd = new int[mMpiData.mProcNum];
        pProcNum = new int[mMpiData.mProcNum];

        for (int i = 0; i < RowNum; i++)
            pProcPivotIter[i] = -1;

        if (mMpiData.mProcRank == 0) {
            pMatrix = new double[mSize * mSize];
            pVector = new double[mSize];
            pResult = new double[mSize];
            RandomDataInitialization(pMatrix, pVector);
        }
    }

    // Function for the data distribution among the processes
    void DataDistribution(double* pMatrix, double* pProcRows, double* pVector, double* pProcVector, int RowNum) {

        int* pSendNum;     // Number of the elements sent to the process
        int* pSendInd;     // Index of the first data element sent 
        // to the process
        int RestRows = mSize; // Number of rows, that have not been 
        // distributed yet
        int i;             // Loop variable

        // Alloc memory for temporary objects
        pSendInd = new int[mMpiData.mProcNum];
        pSendNum = new int[mMpiData.mProcNum];

        // Define the disposition of the matrix rows for the current process
        RowNum = (mSize / mMpiData.mProcNum);
        pSendNum[0] = RowNum * mSize;
        pSendInd[0] = 0;
        for (i = 1; i < mMpiData.mProcNum; i++) {
            RestRows -= RowNum;
            RowNum = RestRows / (mMpiData.mProcNum - i);
            pSendNum[i] = RowNum * mSize;
            pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
        }

        // Scatter the rows
        MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
            pSendNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Define the disposition of the matrix rows for current process
        RestRows = mSize;
        pProcInd[0] = 0;
        pProcNum[0] = mSize / mMpiData.mProcNum;
        for (i = 1; i < mMpiData.mProcNum; i++) {
            RestRows -= pProcNum[i - 1];
            pProcNum[i] = RestRows / (mMpiData.mProcNum - i);
            pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
        }

        MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector,
            pProcNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Free the memory
        delete[] pSendNum;
        delete[] pSendInd;
    }

    // Function for gathering the result vector
    void ResultCollection(double* pProcResult, double* pResult) {
        //Gather the whole result vector on every processor
        MPI_Gatherv(pProcResult, pProcNum[mMpiData.mProcRank], MPI_DOUBLE, pResult,
            pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Fuction for the column elimination
    void ParallelEliminateColumns(double* pProcRows, double* pProcVector, double* pPivotRow, int RowNum, int Iter) {
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
    void ParallelGaussianElimination(double* pProcRows, double* pProcVector, int RowNum) {
        double MaxValue;   // Value of the pivot element of th  process
        int    PivotPos;   // Position of the pivot row in the process stripe 
        // Structure for the pivot row selection
        struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;

        // pPivotRow is used for storing the pivot row and the corresponding 
        // element of the vector b
        double* pPivotRow = new double[mSize + 1];

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
            MPI_Bcast(pPivotRow, mSize + 1, MPI_DOUBLE, Pivot.ProcRank, MPI_COMM_WORLD);

            ParallelEliminateColumns(pProcRows, pProcVector, pPivotRow, RowNum, i);
        }
    }
    // Function for finding the pivot row of the back substitution
    void FindBackPivotRow(int RowIndex, int& IterProcRank, int& IterPivotPos) {
        for (int i = 0; i < mMpiData.mProcNum - 1; i++) {
            if ((pProcInd[i] <= RowIndex) && (RowIndex < pProcInd[i + 1]))
                IterProcRank = i;
        }
        if (RowIndex >= pProcInd[mMpiData.mProcNum - 1])
            IterProcRank = mMpiData.mProcNum - 1;
        IterPivotPos = RowIndex - pProcInd[IterProcRank];
    }

    // Function for the back substitution
    void ParallelBackSubstitution(double* pProcRows, double* pProcVector, double* pProcResult, int RowNum) {
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
    void ParallelResultCalculation(double* pProcRows, double* pProcVector, double* pProcResult, int RowNum) {
        ParallelGaussianElimination(pProcRows, pProcVector, RowNum);
        ParallelBackSubstitution(pProcRows, pProcVector, pProcResult, RowNum);
    }

    // Function for computational process termination
    void ProcessTermination(double* pMatrix, double* pVector, double* pResult, double* pProcRows, double* pProcVector, double* pProcResult) {
        if (mMpiData.mProcRank == 0) {
            delete[] pMatrix;
            delete[] pVector;
            delete[] pResult;
        }
        delete[] pProcRows;
        delete[] pProcVector;
        delete[] pProcResult;

        delete[] pParallelPivotPos;
        delete[] pProcPivotIter;

        delete[] pProcInd;
        delete[] pProcNum;
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

    int* pParallelPivotPos;
    int* pProcPivotIter;
    int* pProcInd;
    int* pProcNum;

    double* pMatrix;        // Matrix of the linear system
    double* pVector;        // Right parts of the linear system
    double* pResult;        // Result vector
    double* pProcRows;      // Rows of the matrix A
    double* pProcVector;    // Block of the vector b
    double* pProcResult;    // Block of the vector x
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
        for (int i = 500; i <= 3000; i += 500) {
            experiment(i);
        }
    }

private:
    void experiment(int size) {
        ParallelGauss gauss(mMpiData, size);

        //auto start = MPI_Wtime();
        //auto start = clock();

        double start, finish, duration;

        // Memory allocation and data initialization
        gauss.ProcessInitialization(gauss.pMatrix, gauss.pVector, gauss.pResult, gauss.pProcRows, gauss.pProcVector, gauss.pProcResult, gauss.RowNum);
        // The execution of the parallel Gauss algorithm
        start = MPI_Wtime();

        gauss.DataDistribution(gauss.pMatrix, gauss.pProcRows, gauss.pVector, gauss.pProcVector, gauss.RowNum);

        gauss.ParallelResultCalculation(gauss.pProcRows, gauss.pProcVector, gauss.pProcResult, gauss.RowNum);

        gauss.ResultCollection(gauss.pProcResult, gauss.pResult);

        finish = MPI_Wtime();
        duration = finish - start;

        gauss.TestResult(gauss.pMatrix, gauss.pVector, gauss.pResult);

        // Printing the time spent by Gauss algorithm
        if (mMpiData.mProcRank == 0)
            printf("\n Time of execution: %f\n", duration);

        // Computational process termination
        gauss.ProcessTermination(gauss.pMatrix, gauss.pVector, gauss.pResult, gauss.pProcRows, gauss.pProcVector, gauss.pProcResult);

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