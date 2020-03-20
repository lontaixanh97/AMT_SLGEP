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

    def sample(self, nos, config):
        # print('nos,self.vars', nos,self.vars)
        nos = int(nos)
        if self.modeltype == 'mvarnorm':
            # solutions = np.random.multivariate_normal(self.mean_true, self.covarmat_true, size=nos)
            solutions = []
            for i in range(nos):
                solution = np.random.multivariate_normal(self.mean_true, self.covarmat_true)
                while not (self.check_solution(solution, config)):
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

