'''
This module is defining the domain for the sampling of reduced order models.
It allows the differentiation between random, Latin hypercube, regular grid,
and Sobol sequence sampling based on the different functions in the object
GridDef:
    .RegularGrid:   Regular grid sampling
    .LHS:           Latin Hypercube sampling
    .Rand:          Random sampling
    .Sobol:         Sobol sequence sampling

Further functions in the object are:
    .scaling:       Scaling of the resulting [0,1] sampling domain to the
                    defined upper and lower bounds. 
    .dep_prop:      Incorporation of proportional dependencies.
    .dep_inv:       Incorporation of inverse proportional dependencies.

The utization of the sampling requires to call first the function GridCase as
for example "case = GridCase()" for the initialization of the grid. Subsequently,
the parameters can be assigned to each case.
The different parameters are then given in the "case" structure.
General parameters:
    .corn_points:   Inclusion of corner points (True or False)
    .lb_m_ub:       Two dimensional array for the lower bound (row 0),
                    nominal value (row 1), and higher bound (row 2) of the
                    independent variables.
    .ind:           One dimensional array with indices of the varied inpendent
                    variables. All other variables in .lb_m_ub are returned at
                    their nominal values.
    .nsamp:         Number of sampling points, per dimension for regular grid
                    and total for LHS and random sampling.
    .dep_prop:      Indices for variables with proportional dependencies.
                    This array has to be two dimensional, with each index pair
                    occupying a row. The first index corresponds to the independent,
                    the second index to the dependent variable.
                    If the dependent variable is varied, then .lb_m_ub has to be
                    adjusted so that the lower and upper bound correspond to the
                    upper and lower bound of the dependency ratio.
    .dep_inv:       Indices for variables with inverse proportional dependencies.
                    This array has to be two dimensional, with each index pair
                    occupying a row. The first index corresponds to the independent,
                    the second index to the dependent variable.
                    If the dependent variable is varied, then .lb_m_ub has to be
                    adjusted so that the lower and upper bound correspond to the
                    upper and lower bound of the dependency ratio.
                    If the dependent variable is not varied, then the lower bound
                    has to equal the dependency ratio.
    .data:          Logic parameter for the decision if data from previous unit
                    operations is provided. If data is True, then further
                    parameters (self.ind_flow) have to be provided.
    .ind_flow:      Indices of the flows which form the polytope. Note that the
                    same indexing as in .ind and .dep_prop is used and the
                    indices to NOT correspond to the indices in the provided
                    data.
    .method:        Chosen sampling method. The available methods are:
                    Regular grid sampling
                    Latin Hypercube sampling
                    Sobol sampling
                    Monte Carlo sampling

Design of experiment specific parameters:
    .crit:          LHS sampling:   Criterion (see PyDOE webpage).
                                    Standard: "maximin"
    .skip:          Sobol sequence: Skip of first x values.
                                    Standard: 0
    .leap:          Sobol sequence: Number of points leaped over for each point.
                                    Standard: 0
'''

# Import of the required packages
import numpy as np
from pyDOE import lhs
from sobol import i4_sobol_generate
from polytope import GridPolytope

class DoECase:
    ''' Class for the initialization of the grid'''

    def __init__(self):
        self.corn_points = False
        self.lb_m_ub  = np.zeros((3,2))
        self.ind      = np.arange(2)
        self.n_samp   = 50
        self.dep_prop = np.empty((0,0))
        self.dep_inv  = np.empty((0,0))
        self.crit     = "maximin"
        self.skip     = 0
        self.leap     = 0
        self.data     = False
        self.ind_flow = ()
        self.method   = 'MonteCarlo'

class DoEDef:
    def __init__(self,case,**kwargs):
        '''
        Base function for the grid definition.

        This function calculates several properties which are required for the
        subsequent sampling domain calculation. These calculations are based on
        the GridCase class. Furthermore, it initialises the sampling domain for
        the subsequent application of the sampling.
        
        If previous data is supposed to be used for the incorporation of
        dependencies, it is necessary to provide as
        additional input a list with the following entries:
            Domain:     Sampling data as outlet from the previous UO
            Ind_flow:   Indices of flowrates which should be investigated
            Ind_ratio:  Indices of ratios which are seen as 
        '''
        self.case     = case
        self.skip     = case.skip
        self.ind      = ()
        self.ind_var  = ()
        self.ind_flow = ()

        # Calculation of the several integers for simplified description
        self.n_var      = np.shape(case.ind)[0]
        self.n_tot      = np.shape(case.lb_m_ub)[1]
        self.n_dep_prop = np.shape(case.dep_prop)[0]
        self.n_dep_inv  = np.shape(case.dep_inv)[0]

        # Calculation of the corner points, if specified
        if case.corn_points:
            r = np.linspace(0.,1.,2)
            self.CP     = np.array(np.meshgrid(*[r]*self.n_var)).T.reshape(-1,self.n_var)
            self.n_samp = case.n_samp+2**self.n_var
        else:
            self.CP     = np.empty((0,self.n_var))
            self.n_samp = case.n_samp

        # Check of and, if true, assignment of sampling data
        if "Data_dict" in kwargs:
            data_dict   = kwargs["Data_dict"]
            domain      = data_dict["Domain"]
            ind_flow    = data_dict["Ind_flow"]
            ind_ratio   = data_dict["Ind_ratio"]
            ind_sum     = data_dict["Ind_sum"]
            self.data = True
            self.poly = GridPolytope()
            self.poly.Polytope_Cal(domain,ind_flow,ind_ratio,ind_sum)
            for k in range(self.n_var):
                if not any(case.ind[k] == case.ind_flow):
                    self.ind      = self.ind+(case.ind[k],)
                    self.ind_var  = self.ind_var+(k,)
                else:
                    self.ind_flow = self.ind_flow+(k,)
        else:
            self.data = False
            self.ind = case.ind
            self.ind_var = tuple(np.arange(self.n_var))

        # Predefinition of the sampling grid with the nominal values
        self.grid   = np.ones((self.n_samp,self.n_tot))*case.lb_m_ub[1,:]

    def DoE(self,case,**kwargs):
        '''
        Function for the calculation and scaling of the sampling domain
        based on the chosen basis function.

        '''

        # Definition of the initial sampling points based on the chosen sampling method
        if case.method == 'RegularGrid':
            # Calculation of the number of sampling points and predefinition of the grid
            self.n_samp = case.n_samp**self.n_var
            self.grid = np.ones((self.n_samp,self.n_tot))*case.lb_m_ub[1,:]

            # Calculation of the regular grid
            r = np.linspace(0.,1.,case.n_samp)
            self.grid_var = np.array(np.meshgrid(*[r]*self.n_var)).T.reshape(-1,self.n_var)
        elif case.method == 'LHS':
            # Calculation of the Latin hypercube
            self.grid_var = np.append(self.CP,lhs(self.n_var,case.n_samp,case.crit),axis = 0)
        elif case.method == 'Sobol':
            # Calculation of the Sobol sequence
            self.grid_var = np.append(self.CP,i4_sobol_generate(self.n_var,case.n_samp,skip = self.skip,leap = case.leap),axis = 0)
            self.skip = self.skip + case.n_samp*(case.leap+1)
        elif case.method == 'MonteCarlo':
            # Calculation of the Monte Carlo sampling
            self.grid_var = np.append(self.CP,np.random.rand(case.n_samp,self.n_var),axis = 0)

        # Scaling of the resulting sampling domain
        scaled_grid_var = self.scaling(case)
        self.grid[:,self.ind] = scaled_grid_var

        # Calculation of the dependencies, including polytope dependency
        if self.n_dep_prop > 0:
            self.dep_prop(case)
        if self.n_dep_inv  > 0:
            self.dep_inv(case)
        if self.data:
            self.grid[:,(case.ind_flow)] = self.poly.Simp_Point(domain = self.grid_var[:,self.ind_flow]) 
        
        scaled_grid = self.grid
        return scaled_grid

    def scaling(self,case):
        '''
        Function for scaling of the resulting grid for the lower and upper
        bounds. The scaling excludes the variables defined via the polytope as
        these variables are already scaled.
        '''
        scaled_grid_var = self.grid_var[:,self.ind_var]*np.diff(case.lb_m_ub[np.ix_((0,2),self.ind)],axis=0)+case.lb_m_ub[0,self.ind]

        return scaled_grid_var

    def dep_prop(self,case):
        '''
        Function for the  incorporation of proportional dependencies between
        pairs of the independent variables.
        '''
        for i in range(self.n_dep_prop):
            # The following if-loop is necessary to be able to incorporate 
            # dependencies even if the dependent variable is not varied, 
            # but only the independent variable.
            if any(case.ind == case.dep_prop[i,1]):
                self.grid[:,case.dep_prop[i,1]] = self.grid[:,case.dep_prop[i,1]]*self.grid[:,case.dep_prop[i,0]]
            else:
                self.grid[:,case.dep_prop[i,1]] = self.grid[:,case.dep_prop[i,0]]*case.lb_m_ub[1,case.dep_prop[i,1]]/case.lb_m_ub[1,case.dep_prop[i,0]]

    def dep_inv(self,case):
        '''
        Function for the incorporation of inverse proporional dependencies
        between pairs of the independent variables.
        '''
        for i in range(self.n_dep_inv):
            # The following if-loop is necessary to be able to incorporate 
            # dependencies even if the dependent variable is not varied, 
            # but only the independent variable.
            if any(case.ind == case.dep_inv[i,1]):
                #self.grid[:,case.dep_inv[i,1]] = case.lb_m_ub[1,case.dep_inv[i,1]]-(self.grid[:,case.dep_inv[i,0]]-case.lb_m_ub[1,case.dep_inv[i,0]])*case.dep_inv[i,2]+self.grid[:,case.dep_inv[i,1]]
                self.grid[:,case.dep_inv[i,1]] = case.lb_m_ub[1,case.dep_inv[i,1]]-(self.grid[:,case.dep_inv[i,0]]-case.lb_m_ub[1,case.dep_inv[i,0]])*self.grid[:,case.dep_inv[i,1]]
            else:
                self.grid[:,case.dep_inv[i,1]] = case.lb_m_ub[1,case.dep_inv[i,1]]-(self.grid[:,case.dep_inv[i,0]]-case.lb_m_ub[1,case.dep_inv[i,0]])*case.dep_inv[i,2]
