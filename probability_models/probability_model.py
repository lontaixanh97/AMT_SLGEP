import numpy as np
from scipy.stats import multivariate_normal
from slgep_lib import *


class ProbabilityModel:  # Works reliably for 2(+) Dimensional distributions
    """ properties
        modeltype; % multivariate normal ('mvarnorm' - for real coded) or univariate marginal distribution ('umd' - for binary coded)    
        mean_noisy;
        mean_true;
        covarmat_noisy;
        covarmat_true;
        probofone_noisy;
        probofone_true;
        probofzero_noisy;
        probofzero_true;    
        vars;
      end"""

    #  methods (Static)
    def __init__(self, modeltype):
        self.modeltype = modeltype

    def check_solution(self, solution, config):
        dims = len(solution)
        for i in range(dims):
            low, high = ChromosomeFactory(config).get_feasible_range(i)
            if low > solution[i] or solution[i] >= high:
                print(".", end='')
                return False
        return True

    def sample(self, nos, config):
        # print('nos,self.vars', nos,self.vars)
        nos = int(nos)
        if self.modeltype == 'mvarnorm':
            # solutions = np.random.multivariate_normal(self.mean_true, self.covarmat_true, size=nos)
            solutions = []
            for i in range(nos):
                solution = np.random.multivariate_normal(self.mean_true, self.covarmat_true)
                while not(self.check_solution(solution, config)):
                    solution = np.random.multivariate_normal(self.mean_true, self.covarmat_true)
                solutions.append(solution.astype(int))
        elif self.modeltype == 'umd':

            solutions = np.random.rand(nos, int(self.vars))
            for i in range(nos):
                index1 = solutions[i, :] <= self.probofone_true
                index0 = solutions[i, :] > self.probofone_true
                solutions[i, index1] = 1
                solutions[i, index0] = 0
        return np.array(solutions)

    def pdfeval(self, solutions):
        """Calculating the probabilty of every solution
        
        Arguments:
            solutions {[2-D Array]} -- [solution or population of evolutionary algorithm]
        
        Returns:
            [1-D Array] -- [probabilty of every solution]
        """

        if self.modeltype == 'mvarnorm':
            # create a multivariate Gaussian object with specified mean and covariance matrix
            # mvn = multivariate_normal(self.mean_noisy, self.covarmat_noisy)
            # print('abc', mvn.shape)
            # probofsols = mvn.pdf(solutions)
            # print(self.mean_noisy)
            # with np.printoptions(threshold=np.inf):
            #     print(self.mean_noisy)
            #     print(np.diag(self.covarmat_noisy1))
            # print(self.covarmat_noisy)
            probofsols = multivariate_normal.pdf(solutions, self.mean_noisy, self.covarmat_noisy1, allow_singular=True)
            # print(probofsols)
        elif self.modeltype == 'umd':
            nos = solutions.shape[0]
            probofsols = np.zeros(nos)
            probvector = np.zeros(self.vars)
            for i in range(nos):
                index = solutions[i, :] == 1
                probvector[index] = self.probofone_noisy[index]
                index = solutions[i, :] == 0
                probvector[index] = self.probofzero_noisy[index]
                probofsols[i] = np.prod(probvector)
        return probofsols

    def buildmodel(self, solutions, config):
        pop, self.vars = solutions.shape
        if self.modeltype == 'mvarnorm':
            self.mean_true = np.mean(solutions, 0)
            # Tính ma trận hiệp phương sai của solutions. Ma trận hiệp phương sai có đường chéo chính là phương sai
            # của các mẫu dữ liệu theo từng chiều
            covariance = np.cov(solutions.T)
            # Simplifying to univariate distribution by ignoring off diagonal terms of covariance matrix
            # Giữ lại đường chéo chính của ma trận hiệp phương sai
            self.covarmat_true = np.diag(np.diag(covariance))
            # Thêm 10% noise để tránh overfit
            pop_noisy = ChromosomeFactory(config).initialize()
            solutions_noisy = np.append(solutions, pop_noisy[:round(0.1 * pop)], 0)
            self.mean_noisy = np.mean(solutions_noisy, 0)
            # print('mean.shape', self.mean_noisy.shape)
            covariance = np.cov(solutions_noisy.T)
            # Simplifying to univariate distribution by ignoring off diagonal terms of covariance matrix
            self.covarmat_noisy1 = np.diag(np.diag(covariance))
            # self.covarmat_noisy = np.cov(solutions_noisy.T)
        elif self.modeltype == 'umd':
            self.probofone_true = np.mean(solutions, 0)
            # print(self.probofone_true)
            self.probofzero_true = 1 - self.probofone_true
            # print('probofone_true')
            # print(self.probofzero_true.shape)
            solutions_noisy = np.append(solutions, np.round(np.random.rand(round(0.1 * pop), self.vars)), axis=0)
            # print(solutions_noisy.shape)
            self.probofone_noisy = np.mean(solutions_noisy, 0)
            # print(self.probofone_noisy)
            self.probofzero_noisy = 1 - self.probofone_noisy
            # print(self.probofzero_noisy)
