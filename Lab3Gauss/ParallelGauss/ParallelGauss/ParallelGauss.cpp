#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <time.h>
#include <math.h>
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

    void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
        int i, j;  // Loop variables
        srand(unsigned(clock()));
        for (i = 0; i < Size; i++) {
            pVector[i] = rand() / double(1000);
            for (j = 0; j < Size; j++) {
                if (j <= i)
                    pMatrix[i * Size + j] = rand() / double(1000);
                else
                    pMatrix[i * Size + j] = 0;
            }
        }
    }

    void ProcessInitialization(double*& pMatrix, double*& pVector,
        double*& pResult, double*& pProcRows, double*& pProcVector,
        double*& pProcResult, int& Size, int& RowNum) {

        int RestRows; // Number of rows, that haven't been distributed yet
        int i;        // Loop variable

        if (mMpiData.mProcRank == 0) {
            do {
                printf("\nEnter the size of the matrix and the vector: ");
                scanf("%d", &Size);
                if (Size < mMpiData.mProcNum) {
                    printf("Size must be greater than number of processes! \n");
                }
            } while (Size < mMpiData.mProcNum);
        }
        MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        RestRows = Size;
        for (i = 0; i < mMpiData.mProcRank; i++)
            RestRows = RestRows - RestRows / (mMpiData.mProcNum - i);
        RowNum = RestRows / (mMpiData.mProcNum - mMpiData.mProcRank);

        pProcRows = new double[RowNum * Size];
        pProcVector = new double[RowNum];
        pProcResult = new double[RowNum];

        pParallelPivotPos = new int[Size];
        pProcPivotIter = new int[RowNum];

        pProcInd = new int[mMpiData.mProcNum];
        pProcNum = new int[mMpiData.mProcNum];

        for (int i = 0; i < RowNum; i++)
            pProcPivotIter[i] = -1;

        if (mMpiData.mProcRank == 0) {
            pMatrix = new double[Size * Size];
            pVector = new double[Size];
            pResult = new double[Size];
            // DummyDataInitialization (pMatrix, pVector, Size);
            RandomDataInitialization(pMatrix, pVector, Size);
        }
    }

    // Function for the data distribution among the processes
    void DataDistribution(double* pMatrix, double* pProcRows, double* pVector,
        double* pProcVector, int Size, int RowNum) {

        int* pSendNum;     // Number of the elements sent to the process
        int* pSendInd;     // Index of the first data element sent 
        // to the process
        int RestRows = Size; // Number of rows, that have not been 
        // distributed yet
        int i;             // Loop variable

        // Alloc memory for temporary objects
        pSendInd = new int[mMpiData.mProcNum];
        pSendNum = new int[mMpiData.mProcNum];

        // Define the disposition of the matrix rows for the current process
        RowNum = (Size / mMpiData.mProcNum);
        pSendNum[0] = RowNum * Size;
        pSendInd[0] = 0;
        for (i = 1; i < mMpiData.mProcNum; i++) {
            RestRows -= RowNum;
            RowNum = RestRows / (mMpiData.mProcNum - i);
            pSendNum[i] = RowNum * Size;
            pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
        }

        // Scatter the rows
        MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
            pSendNum[mMpiData.mProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Define the disposition of the matrix rows for current process
        RestRows = Size;
        pProcInd[0] = 0;
        pProcNum[0] = Size / mMpiData.mProcNum;
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

    // Function for formatted matrix output
    void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
        int i, j; // Loop variables
        for (i = 0; i < RowCount; i++) {
            for (j = 0; j < ColCount; j++)
                printf("%7.4f ", pMatrix[i * ColCount + j]);
            printf("\n");
        }
    }

    // Function for formatted vector output
    void PrintVector(double* pVector, int Size) {
        int i;
        for (i = 0; i < Size; i++)
            printf("%7.4f ", pVector[i]);
    }

    // Function for formatted result vector output
    void PrintResultVector(double* pResult, int Size) {
        int i;
        for (i = 0; i < Size; i++)
            printf("%7.4f ", pResult[pParallelPivotPos[i]]);
    }


    // Fuction for the column elimination
    void ParallelEliminateColumns(double* pProcRows, double* pProcVector, double* pPivotRow, int Size, int RowNum, int Iter) {
        double multiplier;
        for (int i = 0; i < RowNum; i++) {
            if (pProcPivotIter[i] == -1) {
                multiplier = pProcRows[i * Size + Iter] / pPivotRow[Iter];
                for (int j = Iter; j < Size; j++) {
                    pProcRows[i * Size + j] -= pPivotRow[j] * multiplier;
                }
                pProcVector[i] -= pPivotRow[Size] * multiplier;
            }
        }
    }


    // Function for the Gausian elimination
    void ParallelGaussianElimination(double* pProcRows, double* pProcVector, int Size, int RowNum) {
        double MaxValue;   // Value of the pivot element of th  process
        int    PivotPos;   // Position of the pivot row in the process stripe 
        // Structure for the pivot row selection
        struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;

        // pPivotRow is used for storing the pivot row and the corresponding 
        // element of the vector b
        double* pPivotRow = new double[Size + 1];

        // The iterations of the Gaussian elimination stage
        for (int i = 0; i < Size; i++) {

            // Calculating the local pivot row
            double MaxValue = 0;
            for (int j = 0; j < RowNum; j++) {
                if ((pProcPivotIter[j] == -1) && (MaxValue < fabs(pProcRows[j * Size + i]))) {
                    MaxValue = fabs(pProcRows[j * Size + i]);
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
                for (int j = 0; j < Size; j++) {
                    pPivotRow[j] = pProcRows[PivotPos * Size + j];
                }
                pPivotRow[Size] = pProcVector[PivotPos];
            }
            MPI_Bcast(pPivotRow, Size + 1, MPI_DOUBLE, Pivot.ProcRank, MPI_COMM_WORLD);

            ParallelEliminateColumns(pProcRows, pProcVector, pPivotRow, Size, RowNum, i);
        }
    }
    // Function for finding the pivot row of the back substitution
    void FindBackPivotRow(int RowIndex, int Size, int& IterProcRank, int& IterPivotPos) {
        for (int i = 0; i < mMpiData.mProcNum - 1; i++) {
            if ((pProcInd[i] <= RowIndex) && (RowIndex < pProcInd[i + 1]))
                IterProcRank = i;
        }
        if (RowIndex >= pProcInd[mMpiData.mProcNum - 1])
            IterProcRank = mMpiData.mProcNum - 1;
        IterPivotPos = RowIndex - pProcInd[IterProcRank];
    }

    // Function for the back substitution
    void ParallelBackSubstitution(double* pProcRows, double* pProcVector,
        double* pProcResult, int Size, int RowNum) {
        int IterProcRank;    // Rank of the process with the current pivot row
        int IterPivotPos;    // Position of the pivot row of the process
        double IterResult;   // Calculated value of the current unknown
        double val;

        // Iterations of the back substitution stage
        for (int i = Size - 1; i >= 0; i--) {

            // Calculating the rank of the process, which holds the pivot row
            FindBackPivotRow(pParallelPivotPos[i], Size, IterProcRank, IterPivotPos);

            // Calculating the unknown
            if (mMpiData.mProcRank == IterProcRank) {
                IterResult = pProcVector[IterPivotPos] / pProcRows[IterPivotPos * Size + i];
                pProcResult[IterPivotPos] = IterResult;
            }
            // Broadcasting the value of the current unknown
            MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterProcRank, MPI_COMM_WORLD);

            // Updating the values of the vector b
            for (int j = 0; j < RowNum; j++)
                if (pProcPivotIter[j] < i) {
                    val = pProcRows[j * Size + i] * IterResult;
                    pProcVector[j] = pProcVector[j] - val;
                }
        }
    }

    void TestDistribution(double* pMatrix, double* pVector, double* pProcRows,
        double* pProcVector, int Size, int RowNum) {

        if (mMpiData.mProcRank == 0) {
            printf("Initial Matrix: \n");
            PrintMatrix(pMatrix, Size, Size);
            printf("Initial Vector: \n");
            PrintVector(pVector, Size);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < mMpiData.mProcNum; i++) {
            if (mMpiData.mProcRank == i) {
                printf("\nProcRank = %d \n", mMpiData.mProcRank);
                printf(" Matrix Stripe:\n");
                PrintMatrix(pProcRows, RowNum, Size);
                printf(" Vector: \n");
                PrintVector(pProcVector, RowNum);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }


    // Function for the execution of the parallel Gauss algorithm
    void ParallelResultCalculation(double* pProcRows, double* pProcVector,
        double* pProcResult, int Size, int RowNum) {
        ParallelGaussianElimination(pProcRows, pProcVector, Size, RowNum);
        ParallelBackSubstitution(pProcRows, pProcVector, pProcResult, Size, RowNum);
    }

    // Function for computational process termination
    void ProcessTermination(double* pMatrix, double* pVector, double* pResult,
        double* pProcRows, double* pProcVector, double* pProcResult) {
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
    void TestResult(double* pMatrix, double* pVector, double* pResult, int Size) {
        /* Buffer for storing the vector, that is a result of multiplication
           of the linear system matrix by the vector of unknowns */
        double* pRightPartVector;
        // Flag, that shows wheather the right parts vectors are identical or not
        int equal = 0;
        double Accuracy = 1.e-6; // Comparison accuracy

        if (mMpiData.mProcRank == 0) {
            pRightPartVector = new double[Size];
            for (int i = 0; i < Size; i++) {
                pRightPartVector[i] = 0;
                for (int j = 0; j < Size; j++) {
                    pRightPartVector[i] += pMatrix[i * Size + j] * pResult[pParallelPivotPos[j]];
                }
            }

            for (int i = 0; i < Size; i++) {
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

void main(int argc, char* argv[]) {
    double start, finish, duration;
    setvbuf(stdout, 0, _IONBF, 0);

    MPIData mpiData;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiData.mProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiData.mProcNum);

    ParallelGauss gauss(mpiData, 10);

    if (mpiData.mProcRank == 0)
        printf("Parallel Gauss algorithm for solving linear systems\n");

    // Memory allocation and data initialization
    gauss.ProcessInitialization(gauss.pMatrix, gauss.pVector, gauss.pResult, gauss.pProcRows, gauss.pProcVector, gauss.pProcResult, gauss.mSize, gauss.RowNum);
    // The execution of the parallel Gauss algorithm
    start = MPI_Wtime();

    gauss.DataDistribution(gauss.pMatrix, gauss.pProcRows, gauss.pVector, gauss.pProcVector, gauss.mSize, gauss.RowNum);

    gauss.ParallelResultCalculation(gauss.pProcRows, gauss.pProcVector, gauss.pProcResult, gauss.mSize, gauss.RowNum);
    gauss.TestDistribution(gauss.pMatrix, gauss.pVector, gauss.pProcRows, gauss.pProcVector, gauss.mSize, gauss.RowNum);

    gauss.ResultCollection(gauss.pProcResult, gauss.pResult);

    finish = MPI_Wtime();
    duration = finish - start;

    if (mpiData.mProcRank == 0) {
        // Printing the result vector
        printf("\n Result Vector: \n");
        gauss.PrintResultVector(gauss.pResult, gauss.mSize);
    }
    gauss.TestResult(gauss.pMatrix, gauss.pVector, gauss.pResult, gauss.mSize);

    // Printing the time spent by Gauss algorithm
    if (mpiData.mProcRank == 0)
        printf("\n Time of execution: %f\n", duration);

    // Computational process termination
    gauss.ProcessTermination(gauss.pMatrix, gauss.pVector, gauss.pResult, gauss.pProcRows, gauss.pProcVector, gauss.pProcResult);
    MPI_Finalize();
}